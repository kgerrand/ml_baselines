'''
Functions used throughout repo.

'''
#-------------------------------------------------------------

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import joblib
from joblib import load
import calendar
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import config as cfg

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.simplefilter("ignore", InconsistentVersionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

data_path = cfg.path_root/'data_files'

#=======================================================================
def access_model(model_name):
    """
    Accesses the model with the given name

    Args:
    - model_name (str): The name of the model to load

    Returns:
    - model: The loaded model
    """
    model_path = Path('../model_train/model_files') / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"The model file does not exist at path: {model_path}")
 
    else:
        model = joblib.load(model_path)
        return model
    
#=======================================================================
def make_predictions(model, site, compound):
    """
    Make predictions based on the previously trained model, taking into account model type.

    Returns:
    - results (pandas.DataFrame): DataFrame containing the predicted flags, actual flags, and mf values.
    """

    # load in data from baseline_setup.ipynb
    data_balanced_df = pd.read_csv(data_path/'saved_files'/f'for_model_{compound}_{site}.csv', index_col='time')
    data_pca = pd.read_csv(data_path/'saved_files'/f'for_model_pca_{compound}_{site}.csv', index_col='time')
    data_balanced_ds = xr.open_dataset(data_path/'saved_files'/f'data_balanced_ds_{compound}_{site}.nc')

    # removing top three values from index of data_balanced_ds to match the length of the predicted flags
    # this is due to the data balancing process
    data_balanced_ds = data_balanced_ds.isel(time=slice(3, None))

    # making predictions based on model
    # remove predicted_flag if it already exists
    if "predicted_flag" in data_balanced_df.columns:
        data_balanced_df.drop(columns=["predicted_flag"], inplace=True)


    # making predictions based on model type
    model_type = model.__class__.__name__

    # if model is NEURAL NETWORK () - predict normally using meteorological dataset
    if model_type == 'MLPClassifier':
        df_predict = data_balanced_df.copy()
        df_predict.drop(columns=["flag"], inplace=True)
        
        class_probabilities_predict = model.predict_proba(df_predict.reset_index(drop=True))
        threshold = cfg.confidence_threshold
        y_pred = (class_probabilities_predict[:,1] >= threshold).astype(int)
        data_balanced_df["predicted_flag"] = y_pred

    # if model is RANDOM FOREST - predict based on class probabilities using PCA dataset
    if model_type == 'RandomForestClassifier':
        df_predict = data_pca.copy()
        df_predict.drop(columns=["flag"], inplace=True)

        class_probabilities_predict = model.predict_proba(df_predict.reset_index(drop=True))

        threshold = cfg.confidence_threshold
        y_pred = (class_probabilities_predict[:,1] >= threshold).astype(int)

        data_balanced_df["predicted_flag"] = y_pred

    # add mf values to results
    columns_to_keep = ["flag", "predicted_flag"]
    results = data_balanced_df[columns_to_keep].copy()
    results["mf"] = data_balanced_ds.mf.values
    results.index = pd.to_datetime(results.index)

    # removing months with insufficient/missing data
    for year in range(results.index.min().year, results.index.max().year):
        for month in range(1, 13):
            # print(year, month)
            
            # collecting all the data for the given month
            df_month = results.loc[(results.index.year == year) & (results.index.month == month)]
            
            # counting the number of baseline datapoints
            n_baseline_pred = int(df_month["predicted_flag"].sum())

            if n_baseline_pred < 3:
                # dropping month from the dataframe as insufficient data
                results = results.drop(df_month.index)

    return results

#=======================================================================
def quantify_noise(results, compound):
    """
    Quantifies the noise in the true baselines, by calculating the coefficient of variation of the true baseline values based on aggregate data.
    This is a relative measure of dispersion, calculated as the standard deviation divided by the mean, allowing for comparison between different datasets. 
    A higher coefficient of variation indicates a higher level of dispersion.

    Args:
    - results (pandas.DataFrame): Dataframe containing the predicted flags, true flags, and mf values.
    - compound (str): The name of the compound being evaluated.

    Returns:
    - mean_cv (float): The mean coefficient of variation of the true baseline values.
    """

    # extracting true baseline values
    df_actual = results.where(results["flag"] == 1).dropna()
    df_actual.index = pd.to_datetime(df_actual.index)

    # resampling to monthly averages
    df_actual_monthly = df_actual.resample('M').mean()
    df_actual_monthly.index = df_actual_monthly.index.to_period('M')

    # calculating monthly standard deviation
    df_actual_std = df_actual.resample('M').std()
    df_actual_std.index = df_actual_std.index.to_period('M')

    overall_cv = []
    
    # calculating coefficient of variation
    for idx, row in df_actual_std.iterrows():
        cv = row['mf'] / df_actual_monthly.loc[idx, 'mf']
        overall_cv.append(cv)

    # removing nans
    overall_cv = [x for x in overall_cv if str(x) != 'nan']
    mean_cv = np.mean(overall_cv)
    
    return mean_cv

#=======================================================================
def calc_statistics(results):
    """
    Calculates statistics to compare model to true flags.

    Args:
    - results (pandas.DataFrame): Dataframe containing the predicted flags, true flags, and mf values.

    Returns:
    - mae (float): Mean Absolute Error of the model monthly means.
    - rmse (float): Root Mean Squared Error of the model monthly means.
    - mape (float): Mean Absolute Percentage Error of the model monthly means.
    """
    
    # finds mean and standard deviation of mf values for predicted and true baseline values
    actual_values = results["mf"].where((results["flag"] == 1)).dropna()
    predicted_values = results["mf"].where(results["predicted_flag"] == 1).dropna()

    # finds MAE, RMSE and MAPE of model monthly means
    df_pred = results.where(results["predicted_flag"] == 1).dropna()
    df_actual = results.where(results["flag"] == 1).dropna()

    df_pred.index = pd.to_datetime(df_pred.index)
    df_actual.index = pd.to_datetime(df_actual.index)
    df_pred_monthly = df_pred.resample('M').mean()
    df_actual_monthly = df_actual.resample('M').mean()
    df_pred_monthly.index = df_pred_monthly.index.to_period('M')
    df_actual_monthly.index = df_actual_monthly.index.to_period('M')

    mae = np.mean(np.abs(df_pred_monthly["mf"] - df_actual_monthly["mf"]))
    rmse = np.sqrt(np.mean((df_pred_monthly["mf"] - df_actual_monthly["mf"])**2))
    mape = np.mean(np.abs((df_actual_monthly["mf"] - df_pred_monthly["mf"]) / df_actual_monthly["mf"])) * 100

    return (mae, rmse, mape)    

#=======================================================================
def plot_predictions(results, site, compound, model_type):
    """
    Plots mole fraction against time, with the predicted baselines and true baselines highlighted.

    Args:
    - results (pandas.DataFrame): Dataframe containing the predicted flags, true flags, and mf values.
    - site (str): The name of the site.
    - compound (str): The name of the compound being evaluated.
    - model_type (str): The type of model being evaluated.

    Returns:
    - None
    """

    fig, axes = plt.subplots(3,1, figsize=(15,20))
    sns.set_theme(style='ticks', font='Arial')

    # plot 1 - true baselines
    results["mf"].plot(ax=axes[0], label="All Data", color='grey', linewidth=1, alpha=0.5)
    axes[0].scatter(results.index, results["mf"].where(results["flag"] == 1), color='#1ace30', label="NAME/InTEM Baselines", s=2, marker='x')

    # plot 2 - predicted baselines
    results["mf"].plot(ax=axes[1], label="All Data", color='grey', linewidth=1, alpha=0.5)
    axes[1].scatter(results.index, results["mf"].where(results["predicted_flag"] == 1), color='blue', label="Predicted Baselines", s=2, marker='x')

    # plot 3 - comparison
    results["mf"].plot(ax=axes[2], label="All Data", color='grey', linewidth=1, alpha=0.5)
    axes[2].scatter(results.index, results["mf"].where(results["flag"] == 1), color='#1ace30', label="NAME/InTEM Baselines", s=2, marker='x')
    axes[2].scatter(results.index, results["mf"].where(results["predicted_flag"] == 1), color='blue', label="Predicted Baselines", s=2, marker='x')

    # shading the training and validation sets
    if site == 'GSN':
        if results.dropna().index.max() < datetime(2013,1,1):
            pass
        elif results.dropna().index.min() > datetime(2014,12,31):
            pass
        else:
            for ax in axes:
                ax.axvspan(datetime(2013,1,1), datetime(2014,1,1), alpha=0.3, label="Training Set", color='grey')
                ax.axvspan(datetime(2014,1,1), datetime(2014,12,31), alpha=0.2, label="Validation Set", color='purple')

    # all other sites trained on 2018 and validated on 2019
    else:
        if results.dropna().index.max() < datetime(2018,1,1):
            pass
        elif results.dropna().index.min() > datetime(2019,12,31):
            pass
        else:
            for ax in axes:
                ax.axvspan(datetime(2018,1,1), datetime(2019,1,1), alpha=0.3, label="Training Set", color='grey')
                ax.axvspan(datetime(2019,1,1), datetime(2019,12,31), alpha=0.2, label="Validation Set", color='purple')

    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("mole fraction in air / ppt", fontsize=12, fontstyle='italic')

        ax.tick_params(axis='both', which='major', labelsize=10, rotation=0)
        ax.tick_params(axis='both', which='minor', labelsize=8, rotation=0)
        for tick in ax.get_xticklabels():
            tick.set_ha('center')

        ax.legend(loc='best', fontsize=12)


    saving_path = os.path.join('model_results', site, 'plots')
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    fig.savefig(os.path.join(saving_path, f"{model_type.upper()}_{compound}_{site}.png"), dpi=300, bbox_inches='tight')

#=======================================================================
def plot_predictions_monthly(results, site, compound, model_type):
    """
    Plots the predicted baselines and their standard deviations against the true baselines and their standard deviations, highlighting any points outside three standard deviations.

    Args:
    - results (pandas.DataFrame): Dataframe containing the predicted flags, true flags, and mf values.
    - site (str): The name of the site.
    - compound (str): The name of the compound being evaluated.
    - model_type (str): The type of model being evaluated.

    Returns:
    - num_anomalies (int): The number of anomalies detected.
    - anomalies (list): List of anomalous months.
    - num_signif_anomalies (int): The number of significant anomalies detected (greater than 10 standard deviations).
    - signif_anomalies (list): List of significant anomalous months.
    """    

    # extracting flags and predicted flags based on results df
    df_pred = results.where(results["predicted_flag"] == 1).dropna()
    df_actual = results.where(results["flag"] == 1).dropna()

    df_pred.index = pd.to_datetime(df_pred.index)
    df_actual.index = pd.to_datetime(df_actual.index)

    # resampling to monthly averages
    df_pred_monthly = df_pred.resample('M').mean()
    df_actual_monthly = df_actual.resample('M').mean()
    # setting index to year and month only
    df_pred_monthly.index = df_pred_monthly.index.to_period('M')
    df_actual_monthly.index = df_actual_monthly.index.to_period('M')

    # calculating standard deviation
    std_pred_monthly = df_pred.groupby(df_pred.index.to_period('M'))["mf"].std().reset_index()
    std_pred_monthly.set_index('time', inplace=True)
    std_actual_monthly = df_actual.groupby(df_actual.index.to_period('M'))["mf"].std().reset_index()
    std_actual_monthly.set_index('time', inplace=True)


    # plotting
    fig, ax = plt.subplots(figsize=(12,5))
    sns.set_theme(style='ticks', font='Arial')
    ax.minorticks_on()

    df_actual_monthly["mf"].plot(ax=ax, label="True Baselines", color='darkgreen', alpha=0.75, linewidth=1.5)
    if site == 'GSN':
        df_pred_monthly["mf"].plot(ax=ax, label="Predicted Baselines", color='blue', linestyle='--', marker='s', markersize=3, linewidth=1.5)
    else:
        df_pred_monthly["mf"].plot(ax=ax, label="Predicted Baselines", color='blue', linestyle='--', linewidth=1.5)

    # adding standard deviation shading
    upper_actual = df_actual_monthly["mf"] + std_actual_monthly['mf']
    lower_actual = df_actual_monthly["mf"] - std_actual_monthly['mf']

    ax.fill_between(df_actual_monthly.index, lower_actual, upper_actual, color='green', alpha=0.2, label="True Baseline Standard Deviation")

    # Gosan model
    if site == 'GSN':
        if results.dropna().index.max() < datetime(2013,1,1):
            pass
        elif results.dropna().index.min() > datetime(2014,12,31):
            pass
        else:
            ax.axvspan(datetime(2013,1,1), datetime(2014,1,1), alpha=0.3, label="Training Set", color='grey')
            ax.axvspan(datetime(2014,1,1), datetime(2014,12,31), alpha=0.2, label="Validation Set", color='purple')

    # all other sites trained on 2018 and validated on 2019
    else:
        if results.dropna().index.max() < datetime(2018,1,1):
            pass
        elif results.dropna().index.min() > datetime(2019,12,31):
            pass
        else:
            ax.axvspan(datetime(2018,1,1), datetime(2019,1,1), alpha=0.3, label="Training Set", color='grey')
            ax.axvspan(datetime(2019,1,1), datetime(2019,12,31), alpha=0.2, label="Validation Set", color='purple')


    # adding tolerance range based on 3 standard deviations
    upper_range = df_actual_monthly["mf"] + 3*(std_actual_monthly['mf'])
    lower_range = df_actual_monthly["mf"] - 3*(std_actual_monthly['mf'])

    # creating ranges for 5 and 10 standard deviations to quantify anomalies further
    five_upper_range = df_actual_monthly["mf"] + 5*(std_actual_monthly['mf'])
    five_lower_range = df_actual_monthly["mf"] - 5*(std_actual_monthly['mf'])

    ten_upper_range = df_actual_monthly["mf"] + 10*(std_actual_monthly['mf'])
    ten_lower_range = df_actual_monthly["mf"] - 10*(std_actual_monthly['mf'])

    # calculating overall standard deviation for arrows
    overall_std = df_actual_monthly["mf"].std()

    # adding labels to points outside tolerance range
    # looping through in this way as indexes don't always match up (i.e. in the case that no predictions are made in a month)
    anomalous_months = []
    five_std = []
    ten_std = []
    
    for idx, row in df_pred_monthly.iterrows():
        if idx in upper_range.index and row["mf"] >= upper_range.loc[idx]:
            arrow_end = row["mf"] + (overall_std * 0.5)
            ax.annotate(idx.strftime('%B %Y'),
                        xy=(idx, row["mf"]),
                        xytext=(idx, arrow_end),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        horizontalalignment='center', verticalalignment='bottom')
            date = idx.strftime('%Y-%m')
            anomalous_months.append(date)

            if row["mf"] <= five_upper_range.loc[idx]:
                five_std.append(date)
            
            if row["mf"] <= ten_upper_range.loc[idx]:
                ten_std.append(date)
        
        elif idx in upper_range.index and row["mf"] <= lower_range.loc[idx]:
            arrow_end = row["mf"] - (overall_std * 0.5)
            ax.annotate(idx.strftime('%B %Y'),
                        xy=(idx, row["mf"]),
                        xytext=(idx, arrow_end),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        horizontalalignment='center', verticalalignment='bottom')
            date = idx.strftime('%Y-%m')
            anomalous_months.append(date)

            if row["mf"] >= five_lower_range.loc[idx]:
                five_std.append(date)

            if row["mf"] >= ten_lower_range.loc[idx]:
                ten_std.append(date)

    ax.set_ylabel("mole fraction in air / ppt", fontsize=10, fontstyle='italic')
    ax.set_xlabel("")
    ax.legend(loc='best', fontsize=10)

    saving_path = os.path.join('model_results', site, 'plots')
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    fig.savefig(os.path.join(saving_path, f"{model_type.upper()}_{compound}_{site}_monthly.png"), dpi=300, bbox_inches='tight')


    # obtaining anomaly statistics
    num_anomalies = len(anomalous_months)
    num_signif_anomalies = len(ten_std)
    signif_anomalies = ten_std
    anomalies = anomalous_months

    return num_anomalies, anomalies, num_signif_anomalies, signif_anomalies

#=======================================================================
#=======================================================================
def read_intem(site):
    """
    Extracting baseline flags for a given site

    Args:
    - site (str): Site code (e.g., MHD)

    Returns:
    - df (pandas.DataFrame): DataFrame with baseline flags as a binary variable
    """
    
    site_translator = {"MHD":"MH", "CGO":"CG", "GSN":"GS", "JFJ":"J1", "CMN":"M5", "THD":"TH", "ZEP":"ZE", "RPB":"BA", "SMO":"SM"}

    # Filtering so only including data relevant to the given site
    files = (data_path / "manning_baselines").glob(f"{site_translator[site]}*.txt")

    dfs = []

    # Looping through each of the files for the given site
    for file in files:

        # Read the data, skipping metadata, putting into pandas dataframe
        data = pd.read_csv(file, skiprows=6, sep=r'\s+')

        # Setting the index of the dataframe to be the extracted datetime and naming it time
        data.index = pd.to_datetime(data['YY'].astype(str) + "-" + \
                                    data['MM'].astype(str) + "-" + \
                                    data['DD'].astype(str) + " " + \
                                    data['HH'].astype(str) + ":00:00")

        data.index.name = "time"
        
        # Adding the 'Ct' column to the previously created empty list
        dfs.append(data[["Ct"]])
    
    # Creating a dataframe from the list containing all the 'Ct' values
    df = pd.concat(dfs)

    df.sort_index(inplace=True)

    # Replace all values in Ct column less than 10 or greater than 20 with 0
    # not baseline values
    df.loc[(df['Ct'] < 10) | (df['Ct'] >= 20), 'Ct'] = 0

    # Replace all values between 10 and 19 with 1
    # baseline values
    df.loc[(df['Ct'] >= 10) & (df['Ct'] < 20), 'Ct'] = 1

    # Rename Ct column to "baseline"
    df.rename(columns={'Ct': 'baseline'}, inplace=True)

    return df

#=======================================================================
def balance_baselines(ds, minority_ratio): 
    """
    Balances the dataset by randomly undersampling non-baseline data points.

    Args:
    - ds (xarray.Dataset): The dataset to be balanced.
    - minority_ratio (float): The desired ratio of baseline (minority class) data points in the final dataset. 
                            For example, 0.4 means 40% of data points will be baseline.

    Returns:
    - xarray.Dataset: The balanced dataset where the ratio of baseline to non-baseline data points is as specified by the `minority_ratio` argument.

    Raises:
    - ValueError: If the counts of baseline and non-baseline values are not in the expected ratio (within a tolerance of 1%).

    """
    np.random.seed(42)

    # counting number of baseline&non-baseline data points
    baseline_count = ds['baseline'].where(ds['baseline']==1).count()
    non_baseline_count = ds['baseline'].where(ds['baseline']==0).count()
    # print(f"ORIGINAL baseline count: {baseline_count}, non-baseline count: {non_baseline_count}")

    # calculating the minority class count (expected to be baseline)
    minority_count = int(min(baseline_count, non_baseline_count))

    # calculating the majority class count based on majority_ratio and minority_count
    majority_ratio = 1 - minority_ratio
    majority_count = int(minority_count * (majority_ratio/minority_ratio))

    # subsetting the non-baseline data points
    undersampled_non_baseline = ds.where(ds['baseline'] == 0, drop=True)

    # creating an array of time indices & randomly selecting some
    time_indices = undersampled_non_baseline['time'].values
    selected_indices = np.random.choice(time_indices, majority_count, replace=False)
    selected_indices = np.sort(selected_indices)

    # setting the non-baseline data points to only include the randomly selected indices
    undersampled_non_baseline = undersampled_non_baseline.sel(time=selected_indices)

    # combining the the undersampled non-baseline with the baseline values
    balanced_ds = xr.merge([ds.sel(time=(ds['baseline'] == 1)), undersampled_non_baseline])
    balanced_ds = balanced_ds.sortby('time')

    # checking balance
    new_baseline_count = balanced_ds['baseline'].where(balanced_ds['baseline']==1).count()
    new_non_baseline_count = balanced_ds['baseline'].where(balanced_ds['baseline']==0).count()
    # print(f"NEW baseline count: {new_baseline_count}, non-baseline count: {new_non_baseline_count}")

    # verifying that the ratio of baseline:non-baseline data points is as expected (within a tolerance of 1%)
    tolerance = 0.01
    upper_bound = (1+tolerance)*(majority_ratio/minority_ratio)
    lower_bound = (1-tolerance)*(majority_ratio/minority_ratio)

    if(lower_bound <= (new_non_baseline_count/new_baseline_count) <= upper_bound):
        return balanced_ds
    else:
        raise ValueError("The counts of baseline and non-baseline values are not in the expected ratio.")

#=======================================================================
def add_shifted_time(df, points):
    """
    Adds columns with wind data shifted by 6 hours (up three index rows) to the input dataframe.

    Args:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: The dataframe with shifted time columns.
    """

    # copying dataframe
    df_ = df.copy()   

    # extracting wind colunmns
    u10_columns = [f"u10_{point}" for point in points]
    v10_columns = [f"v10_{point}" for point in points]
    u850_columns = [f"u850_{point}" for point in points]
    v850_columns = [f"v850_{point}" for point in points]
    u500_columns = [f"u500_{point}" for point in points]
    v500_columns = [f"v500_{point}" for point in points]
    wind_columns = u10_columns + v10_columns + u850_columns + v850_columns + u500_columns + v500_columns

    # checking if adding a shifted time column has already been done - in which case, remove it before adding it again
    if f'u10_0_past' in df_.columns:
        df_ = df_.drop(columns=[col + f'_past' for col in wind_columns])
        print("Shifted time columns already exist and have been removed. Note that redoing this function will remove additional columns.")

    # create shifted columns
    shifted_columns = [col + '_past' for col in wind_columns]

    # Create a dictionary for the shifted columns
    shifted_dict = {}

    for col, shifted_col in zip(wind_columns, shifted_columns):
        # Shift the column values up by two rows
        shifted_dict[shifted_col] = df_[col].shift(3)

    # Convert the dictionary to a DataFrame
    df_shifted = pd.DataFrame(shifted_dict)

    # Concatenate the original DataFrame with the new DataFrame
    df_ = pd.concat([df_, df_shifted], axis=1)

    # dropping the first three rows as NaN values
    df_ = df_.iloc[3:]

    return df_

#=======================================================================