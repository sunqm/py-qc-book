import yaml
import boto3

ec2_client = boto3.client('ec2') 
ecs_client = boto3.client('ecs')

config = yaml.safe_load('''
family: celery-gpu-worker
networkMode: awsvpc
containerDefinitions:
- name: celery-worker
  image: 987654321000.dkr.ecr.us-east-1.amazonaws.com/python-qc/celery-dft:1.0
  environment:
  - name: OMP_NUM_THREADS
    value: "4"
runtimePlatform:
  cpuArchitecture: X86_64
  operatingSystemFamily: LINUX
cpu: 4 vcpu
memory: 8GB
''')
resp = ecs_client.register_task_definition(**config)
print(yaml.dump(resp))

resp = ec2_client.describe_subnets() 
subnets = [v['SubnetId'] for v in resp['Subnets']] 
config = yaml.safe_load('''
cluster: python-qc
taskDefinition: celery-gpu-worker
count: 2
overrides:
  containerOverrides:
  - name: celery-worker
    environment:
    - name: AWS_ACCESS_KEY_ID
      value: xxxxxxxxxxxxxxxxxxxx
    - name: AWS_SECRET_ACCESS_KEY
      value: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    resourceRequirements:
    - type: GPU
      value: "1"
networkConfiguration: 
  awsvpcConfiguration: 
    subnets: {}
'''.format(subnets))
resp = ecs_client.run_task(**config)
print(yaml.dump(resp))
