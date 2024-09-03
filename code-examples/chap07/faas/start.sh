#!/bin/bash
# Set the default time limit to 1 hour
TIMEOUT=${2:-3600}
S3PATH=$1

aws s3 copy $S3PATH/job.py ./

if (timeout $TIMEOUT python job.py > job.log); then
  python /app/rpc.py finished job.log
  aws s3 copy job.log $S3PATH/
else
  python /app/rpc.py failed job.log
fi

