# Initialise path to root directory - this is where the data and models are stored. Files will be saved relative to this path.
from pathlib import Path
path_root = Path.home()/'OneDrive'/'Kirstin'/'Uni'/'Year4'/'MSciProject'

# Initialise confidence threshold - model will only predict a datapoint as 'baseline' if its confidence is above this threshold
confidence_threshold = 0.8