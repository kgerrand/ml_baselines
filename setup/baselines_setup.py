'''
Variable Setup for Baseline Classification Models - NAME Based Baselines

This script sets up the variables used in the models. 
This involves defining the site and the compound being explored, extracting the meteorological data and reading and balancing the 'true' baselines (obtained from Alaistair Manning at the Met Office). 
The data is then combined to create a dataframe that is saved for use in the models.

'''
#-------------------------------------------------------------
import pandas as pd
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config as cfg
import functions_orig as f

site, compound, _ = f.access_info()
model_type = cfg.model_type

data_path = cfg.path_root/'data_files'

print(f"Setting up variables for {compound} at {site}...")

#-------------------------------------------------------------
# SETTING UP VARIABLES
# extracting baseline flags for given site
df = f.read_intem(site) 
ds_flags = df.to_xarray()

# loading in meteorological data for given site
# 10m wind
ds_10m_u = xr.open_mfdataset((data_path/'meteorological_data'/'ECMWF'/f'{site}_extracted'/'10m_wind_grid').glob('*u*.nc'))
ds_10m_v = xr.open_mfdataset((data_path/'meteorological_data'/'ECMWF'/f'{site}_extracted'/'10m_wind_grid').glob('*v*.nc'))
# 850hPa wind
ds_850hPa_u = xr.open_mfdataset((data_path/'meteorological_data'/'ECMWF'/f'{site}_extracted'/'850hPa_wind_grid').glob('*u*.nc'))
ds_850hPa_v = xr.open_mfdataset((data_path/'meteorological_data'/'ECMWF'/f'{site}_extracted'/'850hPa_wind_grid').glob('*v*.nc'))
# 500hPa wind
ds_500hPa_u = xr.open_mfdataset((data_path/'meteorological_data'/'ECMWF'/f'{site}_extracted'/'500hPa_wind_grid').glob('*u*.nc'))
ds_500hPa_v = xr.open_mfdataset((data_path/'meteorological_data'/'ECMWF'/f'{site}_extracted'/'500hPa_wind_grid').glob('*v*.nc'))
# surface pressure
ds_sp = xr.open_mfdataset((data_path/'meteorological_data'/'ECMWF'/f'{site}_extracted'/'surface_pressure').glob('*.nc'))
# boundary layer height
ds_blh = xr.open_mfdataset((data_path/'meteorological_data'/'ECMWF'/f'{site}_extracted'/'boundary_layer_height').glob('*.nc'))

# loading in AGAGE data for given site and compound
agage_path = data_path / "AGAGE" / "AGAGE-version20250123" / compound
ds_agage = xr.open_dataset(next((agage_path).glob(f"*{site}_{compound}_20250123.nc"))) 
assert ds_agage is not None, 'AGAGE data could not be loaded.'

# creating an xarray dataset with all the meteorological data, the AGAGE data, and the baseline flags, based on the flags time index
# adding a tolerance to the reindexing to allow for the AGAGE data to be reindexed to the nearest hour to avoid extrapolation of missing data
data_ds = xr.merge([ds_flags,
                    ds_10m_u.reindex(time=ds_flags.time, method='nearest'),
                    ds_10m_v.reindex(time=ds_flags.time, method='nearest'),
                    ds_850hPa_u.reindex(time=ds_flags.time, method='nearest'),
                    ds_850hPa_v.reindex(time=ds_flags.time, method='nearest'),
                    ds_500hPa_u.reindex(time=ds_flags.time, method='nearest'),
                    ds_500hPa_v.reindex(time=ds_flags.time, method='nearest'),
                    ds_sp.reindex(time=ds_flags.time, method='nearest'),
                    ds_blh.reindex(time=ds_flags.time, method='nearest'),
                    ds_agage.mf.reindex(time=ds_flags.time, method='nearest', tolerance=np.timedelta64(1, 'h'))],
                    compat='override',
                    combine_attrs='drop',
                    join='override')

data_ds = data_ds.drop_vars('level')
agage_years = np.unique(ds_agage['time.year'])
data_ds = data_ds.sel(time=data_ds['time.year'] <= agage_years[-1]) # dropping any years after the final agage year

data_ds.to_netcdf(data_path/'saved_files'/f'data_ds_{compound}_{site}.nc')

print("Data loaded and merged successfully. \nSetting up for modelling...")

#-------------------------------------------------------------
# BALANCING THE DATASET
minority_ratio = 0.8
using_balanced = True

balanced_data_ds = f.balance_baselines(data_ds, minority_ratio)
balanced_data_ds.to_netcdf(data_path/'saved_files'/f'data_balanced_ds_{compound}_{site}.nc')

#-------------------------------------------------------------
# CREATING THE DATAFRAME FOR MODELLING
data_df = pd.DataFrame({"flag": balanced_data_ds.baseline.values}, index=balanced_data_ds.time.values)

points = balanced_data_ds.points.values

u10_columns = [f"u10_{point}" for point in points]
v10_columns = [f"v10_{point}" for point in points]
u850_columns = [f"u850_{point}" for point in points]
v850_columns = [f"v850_{point}" for point in points]
u500_columns = [f"u500_{point}" for point in points]
v500_columns = [f"v500_{point}" for point in points]

# concatenating the dataframe with the meteorological & temporal data
data_df = pd.concat([
    data_df,
    pd.DataFrame(balanced_data_ds.u10.sel(points=points).values, columns=u10_columns, index=data_df.index),
    pd.DataFrame(balanced_data_ds.v10.sel(points=points).values, columns=v10_columns, index=data_df.index),
    pd.DataFrame(balanced_data_ds.u850.sel(points=points).values, columns=u850_columns, index=data_df.index),
    pd.DataFrame(balanced_data_ds.v850.sel(points=points).values, columns=v850_columns, index=data_df.index),
    pd.DataFrame(balanced_data_ds.u500.sel(points=points).values, columns=u500_columns, index=data_df.index),
    pd.DataFrame(balanced_data_ds.v500.sel(points=points).values, columns=v500_columns, index=data_df.index),
    pd.DataFrame({"sp": balanced_data_ds.sp.values}, index=data_df.index),
    pd.DataFrame({"blh": balanced_data_ds.blh.values}, index=data_df.index),
    pd.DataFrame({"time_of_day": data_df.index.hour}, index=data_df.index),
    pd.DataFrame({"day_of_year": data_df.index.dayofyear}, index=data_df.index)],
    axis=1)

data_df = f.add_shifted_time(data_df, points)
data_df.index.name = "time"

data_df.to_csv(data_path/'saved_files'/f"for_model_{compound}_{site}.csv")

#-------------------------------------------------------------
# DIMENSIONALITY REDUCTION - PCA
if model_type == 'rf':
    data_for_pca = data_df.drop(columns='flag')

    # redefining column name lists to include shifted wind data
    u10_columns = [col for col in data_for_pca.columns if 'u10' in col]
    v10_columns = [col for col in data_for_pca.columns if 'v10' in col]
    u850_columns = [col for col in data_for_pca.columns if 'u850' in col]
    v850_columns = [col for col in data_for_pca.columns if 'v850' in col]
    u500_columns = [col for col in data_for_pca.columns if 'u500' in col]
    v500_columns = [col for col in data_for_pca.columns if 'v500' in col]

    wind_columns = u10_columns + v10_columns + u850_columns + v850_columns + u500_columns + v500_columns

    # standardising the data for PCA based on column groups
    # groups = wind (groups by direction and height), sp, blh, time_of_day, day_of_year
    column_groups = {
        'u10': u10_columns,
        'v10': v10_columns,
        'u850': u850_columns,
        'v850': v850_columns,
        'u500': u500_columns,
        'v500': v500_columns,
        'sp': ['sp'],
        'blh': ['blh'],
        'time_of_day': ['time_of_day'],
        'day_of_year': ['day_of_year']
    }

    # creating dictionary to hold standardised data
    standardised_data = {}

    # standardising each group of columns
    for group, columns in column_groups.items():
        data = data_for_pca[columns]
        
        # reshape if only one column - applicable for all but the wind columns
        if data.shape[1] == 1:
            data = data.values.reshape(-1, 1)
        
        standardised_data[group] = StandardScaler().fit_transform(data)


    # concatenating the standardised data into a dataframe for use in PCA
    # first converting the standardised data into dataframes so dimensions are correct for concatenation
    dfs = [pd.DataFrame(data, columns=columns) for group, columns, data in zip(column_groups.keys(), column_groups.values(), standardised_data.values())]

    # concatenating the dataframes
    data_for_pca = pd.concat(dfs, axis=1)
    data_for_pca.index = data_df.index

    # fitting the PCA with the standardised data
    desired_explained_variance = 0.85
    pca = PCA()
    pca.fit(data_for_pca)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    desired_components = np.argmax(cumulative_variance >= desired_explained_variance) + 1

    # fitting the PCA with the desired number of components for 85% explained variance
    num_components = desired_components
    pca = PCA(n_components=num_components)

    pca_data = pca.fit_transform(data_for_pca)
    pca_components = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(num_components)], index=data_df.index)
    pca_components.sample(5)

    # retrieving the loading for each component
    loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(num_components)], index=data_for_pca.columns)

    # saving the loadings to a csv file
    loadings.to_csv(data_path/'saved_files'/f"pca_loadings_{compound}_{site}.csv")

    # adding the flag column back in and saving the dataframe for use in the model
    pca_components['flag'] = data_df['flag']

    # saving the PCA components dataframe
    pca_components.to_csv(data_path/'saved_files'/f'for_model_pca_{compound}_{site}.csv', index=True)


#-------------------------------------------------------------
print("Variable setup complete.")
print("")
#-------------------------------------------------------------
# END OF SCRIPT
