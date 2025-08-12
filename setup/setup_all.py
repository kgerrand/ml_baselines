'''
This script iterates through all AGAGE sites and compounds, setting up the necessary variables for model training and evaluation.
'''
#-------------------------------------------------------------

import subprocess, sys
sys.path.append('../')
import config as cfg

sites = cfg.agage_sites
compounds = cfg.compounds

for site in sites:
    print(f"\n------ Setting up variables for {site} ------")
    for compound in compounds:
        subprocess.run([
            sys.executable,
            "baselines_setup.py",
            "--site", site,
            "--compound", compound
        ], check=True)

#-------------------------------------------------------------