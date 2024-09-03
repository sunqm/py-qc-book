import boto3
import jinja2
import yaml

CLUSTER = 'pychem-test'
TASK = 'dft-demo'

task_config_tpl = jinja2.Template('''
cluster: {{ cluster }}
taskDefinition: {{ task }}
count: 1
overrides:
  containerOverrides:
  - name: {{ task }}-1
    command:
    - /app/start.sh
    - s3://python-qc-3vtl/{{ task }}/{{ job_id }}
    - "{{ timeout or 7200 }}"
    environment:
    - name: JOB_ID
      value: "{{ job_id }}"
    - name: OMP_NUM_THREADS
      value: "{{ threads or 1 }}"
  cpu: {{ threads or 1 }}
{%- if threads %}
  memory: {{ threads * 2 }}GB
{%- else %}
  memory: 2GB
{%- endif %}
launchType:
  EC2
networkConfiguration:
  awsvpcConfiguration:
    subnets:
{%- for v in subnets %}
    - {{ v }}
{%- endfor %}
''')

ec2_client = boto3.client('ec2')
resp = ec2_client.describe_subnets()
subnets = [v['SubnetId'] for v in resp['Subnets']]

config = task_config_tpl.render(
    cluster=CLUSTER, task=TASK, job_id='001', subnets=subnets)
config = yaml.safe_load(config)

ecs_client = boto3.client('ecs')
print(config)
resp = ecs_client.run_task(**config)
print(yaml.dump(resp))
