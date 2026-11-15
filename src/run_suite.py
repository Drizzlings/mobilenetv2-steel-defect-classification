"""Finite experiment queue with automatic next-run launch and 2-hour checks.

Run in tmux/nohup. This is a supervisor for this finite batch, not a perpetual
cron job: its heartbeat stops automatically when all experiments finish.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from data import ROOT, write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',default='0');a=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    cfg=json.loads((ROOT/'configs'/'config.json').read_text());output=ROOT/'outputs';output.mkdir(exist_ok=True)
    lock=(output/'suite.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another suite supervisor is already active')
    status={'state':'running','pid':os.getpid(),'gpu':a.gpu,'started_unix':time.time(),'current':None,'completed':[]}
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}
    write_json(output/'source_hashes.json',source)
    stop=threading.Event()
    def snapshot(reason):
        event={'time':datetime.now(timezone.utc).isoformat(),'reason':reason,**status}
        event['runs']={p.parent.name:json.loads(p.read_text()) for p in (output/'runs').glob('*/status.json')}
        with (output/'progress_checks.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
        write_json(output/'suite_status.json',event)
    def heartbeat():
        while not stop.wait(cfg['heartbeat_seconds']):snapshot('scheduled_2_hour_check')
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start();snapshot('start')
    try:
        for action in ['train','evaluate']:
            for model in cfg['models']:
                for seed in cfg['seeds']:
                    name=f'{model}_seed{seed}';status['current']={'action':action,'run':name};snapshot('start_stage')
                    command=[sys.executable,'-u',str(ROOT/'src'/'experiment.py'),action,'--model',model,'--seed',str(seed)]
                    if action=='train':command.append('--resume')
                    logpath=output/f'{name}_{action}.log'
                    for attempt in range(2):
                        with logpath.open('a',encoding='utf-8') as logfile:
                            logfile.write(f'\nATTEMPT {attempt+1} {datetime.now(timezone.utc).isoformat()}\n');logfile.flush()
                            code=subprocess.call(command,cwd=ROOT,stdout=logfile,stderr=subprocess.STDOUT)
                        if code==0:break
                        snapshot(f'failure_attempt_{attempt+1}')
                    if code!=0:raise RuntimeError(f'{action} failed twice: {name}; inspect {logpath.name}')
                    status['completed'].append({'action':action,'run':name});snapshot('stage_complete_next_launches_immediately')
        status['state']='complete';status['current']=None;status['finished_unix']=time.time();snapshot('all_experiments_complete')
    except BaseException as error:
        status['state']='failed';status['error']=str(error);snapshot('failed');raise
    finally:stop.set();thread.join(timeout=2)

if __name__=='__main__':main()
