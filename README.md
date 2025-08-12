# ml_baselines
A machine learning library for the estimation of greenhouse gas baseline timeseries from high-frequency observations.

## Running the Code
1. Variable Setup
To run the code, the required dataset must first be created using [baseline_setup.py](https://github.com/kgerrand/ml_baselines/blob/main/setup/baselines_setup.py). 
This collects the relevant meteorology, concentration data and baseline flags. 
The meteorological data were taken from the [EMCWF ERA5 reanalyses](https://cds.climate.copernicus.eu/#!/search?text=ERA5&type=dataset&keywords=((%20%22Product%20type:%20Reanalysis%22%20)%20AND%20(%20%22Variable%20domain:%20Atmosphere%20(surface)%22%20)%20AND%20(%20%22Spatial%20coverage:%20Global%22%20)%20AND%20(%20%22Temporal%20coverage:%20Past%22%20)%20AND%20(%20%22Provider:%20Copernicus%20C3S%22%20))), and the concentration from [AGAGE](https://www-air.larc.nasa.gov/missions/agage/data/version-history/20250123).
This step can be ran for all sites and compounds by running [setup_all.py](https://github.com/kgerrand/ml_baselines/blob/main/setup/setup_all.py).

2. Model Training
The models are trained using the dataset described above. There is a model per site per algorithm (neural network MLP and random forest). The final models are saved and can be found in the [final models](https://github.com/kgerrand/ml_baselines/tree/main/models/model_files) folder. [Summary statistics](https://github.com/kgerrand/ml_baselines/blob/main/model_train/model_stats.csv) are also available.
This step can be ran for all models by running [train_all.py](https://github.com/kgerrand/ml_baselines/blob/main/model_train/train_all.py).

3. Model Evaluation
The models are tested through [quantitative and qualitative evaluation](https://github.com/kgerrand/ml_baselines/blob/main/model_eval/model_eval.py). A chosen subsample of trace species were evaluated, as defined in the [configuration file](https://github.com/kgerrand/ml_baselines/blob/main/config.py). The model outcomes for this species subsample are saved for each site (e.g. [neural network results at Mace Head, Ireland](https://github.com/kgerrand/ml_baselines/blob/main/model_eval/model_results/MHD/MHD_nn.csv)).
This step can be ran for all sites and compounds by running [eval_all.py](https://github.com/kgerrand/ml_baselines/blob/main/model_eval/eval_all.py).