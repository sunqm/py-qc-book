FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pyscf gpu4pyscf-cuda11x boto3 celery[sqs]

COPY dft_app.py /app/
WORKDIR /app
CMD ["celery", "-A", "dft_app", "worker", "--loglevel=info"]
