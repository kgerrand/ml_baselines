# ml_baselines
A machine learning library for the estimation of greenhouse gas baseline timeseries from high-frequency observations.

## Running the Code
1. Variable Setup
2. L

To run the code, the required dataset must first be created using [baseline_setup.py](https://github.com/kgerrand/ml_baselines/blob/main/setup/baselines_setup.py). This collects the relevant meteorology, concentration data and baseline flags. 
The meteorological data were taken from the [EMCWF ERA5 reanalyses](https://cds.climate.copernicus.eu/#!/search?text=ERA5&type=dataset&keywords=((%20%22Product%20type:%20Reanalysis%22%20)%20AND%20(%20%22Variable%20domain:%20Atmosphere%20(surface)%22%20)%20AND%20(%20%22Spatial%20coverage:%20Global%22%20)%20AND%20(%20%22Temporal%20coverage:%20Past%22%20)%20AND%20(%20%22Provider:%20Copernicus%20C3S%22%20))), and the concentration from [AGAGE](https://www-air.larc.nasa.gov/missions/agage/data/version-history/20250123).
Following the creation of the dataset, the models (as defined in [final models](https://github.com/kgerrand/ml_baselines/tree/main/models/model_files)) are tested through [quantitative and qualitative evaluation](https://github.com/kgerrand/ml_baselines/blob/main/model_eval/model_eval.py). 

For simplicity, [setup_all.py](https://github.com/kgerrand/ml_baselines/blob/main/setup/setup_all.py), [train_all.py](https://github.com/kgerrand/ml_baselines/blob/main/model_train/train_all.py), and [eval_all.py](https://github.com/kgerrand/ml_baselines/blob/main/model_eval/eval_all.py) will setup all datasets, train all models and then evaluate them with minimal human intervention required. This process loops through each compound of a chosen subsample defined in [config.py](https://github.com/kgerrand/ml_baselines/blob/main/config.py).