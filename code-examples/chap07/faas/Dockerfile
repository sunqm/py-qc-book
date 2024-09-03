FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pyscf gpu4pyscf awscli boto3 jinja2 yaml requests

COPY start.sh rpc.py pes_scan.py /app/
RUN chmod +x /app/start.sh
WORKDIR /app

