import json
import hashlib
import boto3

# Deploy this function in AWS Lambda service

def lambda_handler(event, context):
    # The data structure of event can be found in
    # https://docs.aws.amazon.com/lambda/latest/dg/services-apigateway.html
    method = event['httpMethod']
    if method != 'POST':
        return {'statusCode': 400}

    body = json.loads(event['body'])
    job = body['job']
    if job.upper() == 'PES':
        config = json.dumps({
            'molecule': body['molecule'],
            'params': body['params']}
        ).encode()
        job_id = hashlib.md5(config).hexdigest()

        s3_client = boto3.client('s3')
        bucket, path = 'python-qc', f'pes-faas/{job_id}'
        s3_client.put_object(Bucket=bucket, Key=f'{path}/config.json', Body=config)

        s3path = f's3://{bucket}/{path}/config.json'
        ecs_client = boto3.client('ecs')
        try:
            resp = ecs_client.run_task(
                cluster='python-qc',
                taskDefinition='pes-rpc-server',
                count=1,
                launchType='FARGATE',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': ['subnet-e529ab30']
                    }
                },
                overrides={
                    'containerOverrides': [{
                        'name': 'rpc-server',
                        'command': ['python', 'pes_scan.py', s3path]
                    }]
                }
            )
        except Exception as e:
            msg = str(e)
        else:
            msg = json.dumps(resp)
        results = f's3://{bucket}/pes-faas/{job_id}/results.log'
    else:
        msg = f'Unknown job {job}'
        results = 'N/A'

    return {
        'statusCode': 200,
        'body': {
            'results': results,
            'detail': msg,
        }
    }
