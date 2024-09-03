import tempfile
import pyscf
from gpu4pyscf.dft import RKS
from celery import Celery
import boto3

app = Celery('dft-pes', broker='sqs://')
s3 = boto3.client('s3')

@app.task
def dft_energy(molecule, s3_path):
    output = tempfile.mktemp()
    mol = pyscf.M(atom=molecule, basis='def2-tzvp', verbose=4, output=output)
    mf = RKS(mol, xc='wb97x').density_fit().run()
    bucket, key = s3_path.replace('s3://', '').split('/', 1)
    s3.upload_file(output, bucket, key)

