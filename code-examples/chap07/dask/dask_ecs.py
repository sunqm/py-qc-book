import yaml
import boto3

ec2_client = boto3.client('ec2') 
resp = ec2_client.describe_subnets() 
subnets = [v['SubnetId'] for v in resp['Subnets']] 
private_ip = ec2_client.describe_instances(
        Filters=[{'Name': 'tag:Name', 'Values': ['dask-server']}]
)['Reservations'][0]['Instances'][0]['PrivateIpAddress']

config = yaml.safe_load('''
cluster: python-qc
taskDefinition: celery-gpu-worker
count: 2
overrides:
  containerOverrides:
  - name: dask-gpu-worker
    image: 987654321000.dkr.ecr.us-east-1.amazonaws.com/python-qc/dask-dft:1.0
    command:
    - dask-worker
    - "{0}:port"
    resourceRequirements:
    - type: GPU
      value: "1"
networkConfiguration: 
  awsvpcConfiguration: 
    subnets: {1}
'''.format(private_ip, subnets}))
ecs_client = boto3.client('ecs')
resp = ecs_client.run_task(**config)
print(yaml.dump(resp))
