import os
import sys
from xmlrpc.client import ServerProxy

rpc_server = os.getenv('RPC_SERVER')
job_id = os.getenv('JOB_ID')
status = sys.argv[1]
logfile = sys.argv[2]

def parse_log(logfile):
    '''Reads the log file and finds the energy'''
    energy = open(logfile, 'r').readline(-1).split(' = ')[1]
    return energy

with ServerProxy(rpc_server) as proxy:
    if status == 'finished':
        proxy.set_result(job_id, parse_log(logfile))
    else:
        proxy.set_result(job_id, status)
