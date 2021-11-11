import subprocess
import requests
import time
import os

print("Start!")

sh_folder = "/cds/home/n/npowersr/XAS/ChemRIXS/rixlw1019/results/Dougie/preproc/v2/"
sh_file = "sbatch_v2.sh"

def is_run_saved(run, exp):
    try:
        location = "SLAC"
        url = f"https://pswww.slac.stanford.edu/ws/lgbk/lgbk/{exp}/ws/{run}/files_for_live_mode_at_location"
        r = requests.get(url, params={"location": location})
        data = r.json()
        if data['success'] and data['value']['all_present'] and data['value']['is_closed']:
            return True
        else:
            return False
    except: #if it doesn't exist it throws an error
        return False
    
def submit_bjob(run, sh_file=sh_file, sh_folder=sh_folder):
    
    cmd = [sh_folder+sh_file, str(run)] # removed the 'sbatch' here because it weirdly started submitting to 
    # psanagpuq on 20210510
    call = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, cwd=sh_folder)
    print(call.stdout)

exp='rixlw1019' #changeme
run = 14 #changeme
run_stop = 34 #changeme

print("Now looking for run %d" % run, end='')
while run <= run_stop:
    if is_run_saved(run, exp):
        print('\n run %i good!\n'%run, flush=True)
        submit_bjob(run)
        run += 1
        print("Now looking for run %d" % run, end='', flush=True)
    else:
        print(".", end='', flush=True)
        time.sleep(10)
