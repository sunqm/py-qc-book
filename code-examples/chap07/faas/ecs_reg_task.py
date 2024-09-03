import boto3
import jinja2
import yaml

TASK = 'dft-demo'

task_config_tpl = jinja2.Template('''
task_definition:
  family: {{ task }}
  networkMode: awsvpc
  containerDefinitions:
    - name: {{ task }}-1
      image: 987654321000.dkr.ecr.us-east-1.amazonaws.com/python-qc/dft-gpu:1.0
      cpu: 4
      memory: 2048
      resourceRequirements:
      - type: GPU
        value: "1"
  runtimePlatform:
    cpuArchitecture: X86_64
    operatingSystemFamily: LINUX
  requiresCompatibilities:
    - EC2
  memory: "4GB"
  cpu: "4"
''')

config = task_config_tpl.render(task=TASK)
config = yaml.safe_load(config)

ecs_client = boto3.client('ecs')
resp = ecs_client.register_task_definition(**config['task_definition'])
print(yaml.dump(resp))
