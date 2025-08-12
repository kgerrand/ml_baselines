# Initialise path to root directory - this is where the data and models are stored. Files will be saved relative to this path.
from pathlib import Path
path_root = Path.home()/'OneDrive'/'Kirstin'/'Uni'/'Year4'/'MSciProject'
def get_path_root():
    print("Please update the path_root variable in config.py to point to the root directory.")
    return Path.home() / ''
path_root = get_path_root()

# Initialise confidence threshold - model will only predict a datapoint as 'baseline' if its confidence is above this threshold
confidence_threshold = 0.8

#-------------------------------------------------------------
# Initialise subsample of compounds to use for model training and evaluation
compounds = ['CH4', 'CF4', 'CFC-12', 'CH2Cl2', 'CH3Br',
            'HCFC-22', 'HFC-125', 'HFC-134a', 'N2O', 'SF6']

# Initialise AGAGE sites to use
agage_sites = ['CGO', 'CMN', 'GSN', 'JFJ', 'MHD', 'RPB', 'SMO', 'THD', 'ZEP']
