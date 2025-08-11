'''
This script iterates through all AGAGE sites and evaluates model performance.
'''
#-------------------------------------------------------------
import subprocess, sys

sites = ["CGO", "CMN", "GSN",
         "JFJ", "MHD", "RPB",
         "SMO", "THD", "ZEP"]
model_types = ["NN", "RF"]

for site in sites:
    for model_type in model_types:
        print(f"\n------ Evaluating {model_type} for {site} ------")
        subprocess.run([
            sys.executable,
            "model_eval.py",
            "--site", site,
            "--model_type", model_type
        ], check=True)

#-------------------------------------------------------------