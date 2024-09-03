import os
import sys
import boto3
import hashlib
import requests
import jinja2
import json
import yaml
from concurrent.futures import Future
from contextlib import contextmanager
from threading import Thread
from xmlrpc.server import SimpleXMLRPCServer

CLUSTER = 'python-qc'
TASK = 'dft-demo'
RPC_PORT = 5005
s3_client = boto3.client('s3')
ecs_client = boto3.client('ecs')

rpc_config_tpl = jinja2.Template('''
cluster: {{ cluster }}
taskDefinition: {{ task }}
count: 1
overrides:
  containerOverrides:
  - name: {{ task }}-1
    command:
    - /app/start.sh
    - {{ job_path }}
    - {{ timeout or 7200 }}
    environment:
    - name: JOB_ID
      value: "{{ job_id }}"
    - name: RPC_SERVER
      value: "{{ rpc_server }}"
    - name: OMP_NUM_THREADS
      value: "{{ threads }}"
    resourceRequirements:
    - type: GPU
      value: "1"
  cpu: {{ threads }} vcpu
  memory: {{ threads * 2 }}GB
launchType:
  EC2
networkConfiguration:
  awsvpcConfiguration:
    subnets:
{%- for v in subnets %}
    - {{ v }}
{%- endfor %}
''')

rpc_job_tpl = jinja2.Template('''
import pyscf
from gpu4pyscf.dft import RKS
mol = pyscf.M(atom="""{{ geom }}""", basis='def2-tzvp', verbose=4)
mf = RKS(mol, xc='b3lyp').density_fit().run()
''')

job_pool = {}

def set_result(job_id, result):
    fut = job_pool[job_id]
    fut.set_result(result)

def geometry_on_pes(molecule, pes_params, existing_results=None):
    '''Generates molecule geometry for the important grids on PES

    Arguments:
    - molecule: Molecular formula
    - pes_params: Targets to scan, such as bonds, bond angles.
    '''
    # This function should produces a set of molecule configurations and refine
    # the configurations based on the results of accomplished calculations.
    # Here is a fake generator to mimic the functionality.
    if existing_results:
        return []
    h2o_xyz = 'O 0 0 0; H 0.757 0.587 0; H -0.757 0.587'
    return [h2o_xyz] * 3

def self_Ip():
    '''IP address of the current container or EC2 instance'''
    # https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-metadata-endpoint-v3.html
    metadata_uri = os.getenv('ECS_CONTAINER_METADATA_URI')
    if metadata_uri:
        resp = requests.get(f'{metadata_uri}/task').json()
        ip = resp['Networks'][0]['IPv4Addresses']
    else:
        # Local IP address for an EC2 instance.
        # https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html
        ip = requests.get('http://169.254.169.254/latest/meta-data/local-ipv4')
    return ip

def launch_tasks(geom_grids, results_path, timeout=7200):
    ip = self_Ip()

    jobs = {}
    for geom in geom_grids:
        job_conf = rpc_job_tpl.render(geom=geom).encode()
        job_id = hashlib.md5(job_conf).hexdigest()
        bucket, key = results_path.replace('s3://', '').split('/', 1)
        job_path = f's3://{bucket}/{key}/{job_id}'
        s3_client.put_object(Bucket=bucket, Key=f'{key}/{job_id}', Body=job_conf)

        task_config = rpc_config_tpl.render(
            cluster=CLUSTER, task=TASK, job_id=job_id, job_path=job_path,
            rpc_server=f'{ip}:{RPC_PORT}', timeout=timeout, threads=1)
        try:
            ecs_client.run_task(**yaml.safe_load(task_config))
        except Exception:
            pass
        else:
            fut = Future()
            fut._timeout = timeout
            job_pool[job_id] = fut
            jobs[job_id] = fut
    return jobs

def parse_config(config_file):
    assert config_file.startswith('s3://')
    bucket, key = config_file.replace('s3://', '').split('/', 1)
    config = json.loads(s3_client.get_object(Bucket=bucket, Key=key))
    return config

def pes_app(config_file):
    config = parse_config(config_file)
    molecule = config['molecule']
    pes_params = config['params']
    results_path = config_file.rsplit('/', 1)[0]

    with rpc_service([set_result]):
        # Scan geometry until enough data are generated
        results = {}
        geom_grids = geometry_on_pes(molecule, pes_params, results)
        while geom_grids:
            jobs = launch_tasks(geom_grids, results_path)
            for key, fut in jobs.items():
                try:
                    result = fut.result(fut._timeout)
                except TimeoutError:
                    result = 'timeout'
                results[key] = result
            geom_grids = geometry_on_pes(molecule, pes_params, results)
    return results

@contextmanager
def rpc_service(funcs):
    '''Creates an RPC service in background'''
    try:
        rpc_server = SimpleXMLRPCServer(("", RPC_PORT))
        for fn in funcs:
            rpc_server.register_function(fn, fn.__name__)
        rpc_service = Thread(target=rpc_server.serve_forever)
        rpc_service.start()
        yield
    finally:
        # Terminate the RPC service
        SimpleXMLRPCServer.shutdown(rpc_server)
        rpc_service.join()

if __name__ == '__main__':
    pes_app(sys.argv[1])
