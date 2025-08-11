'''
Random Forest Final Model - Cape Grim, Australia (CGO)
'''
#-------------------------------------------------------------
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import sys, os
sys.path.append('../../')
import config as cfg

path_root = cfg.path_root
data_path = path_root/'data_files'/'saved_files'

site = 'CGO'
compound = 'cf4'

print(f"Creating a random forest model for {site}...")

#-------------------------------------------------------------
# LOADING DATA
data = pd.read_csv(data_path/f'for_model_pca_{compound}_{site}.csv', parse_dates=['time'])
train_data = data[(data['time'].dt.year >= 2018) & (data['time'].dt.year <= 2018)]
val_data = data[(data['time'].dt.year >= 2019) & (data['time'].dt.year <= 2019)]
test_data = data[~data.index.isin(train_data.index) & ~data.index.isin(val_data.index)]

train_range = f"{train_data['time'].min().date()},{train_data['time'].max().date()}"
val_range = f"{val_data['time'].min().date()},{val_data['time'].max().date()}"
test_range = f"{test_data['time'].min().date()},{test_data['time'].max().date()}"
train_len, val_len, test_len = len(train_data), len(val_data), len(test_data)

# Drop the "time" column as it won't be used in the model
train_data = train_data.drop(columns=['time'])
val_data = val_data.drop(columns=['time'])
test_data = test_data.drop(columns=['time'])

# Define the features (X) and the target (y)
X_train = train_data.drop(columns=['flag'])
y_train = train_data['flag']
X_val = val_data.drop(columns=['flag'])
y_val = val_data['flag']
X_test = test_data.drop(columns=['flag'])
y_test = test_data['flag']

# Balanced Data - removing NaN values and associated data
y_train = y_train.dropna()
y_val = y_val.dropna()
y_test = y_test.dropna()

# aligning indices of features sets
X_train = X_train.loc[y_train.index]
X_val = X_val.loc[y_val.index]
X_test = X_test.loc[y_test.index]

#-------------------------------------------------------------
# CREATING THE RANDOM FOREST MODEL
rf_model = RandomForestClassifier(random_state=42,
                                  n_estimators=100,
                                  max_depth=5,
                                  criterion='entropy',
                                  bootstrap=False,)

rf_model.fit(X_train, y_train)

class_probabilities_val = rf_model.predict_proba(X_val)
class_probabilites_train = rf_model.predict_proba(X_train)

confidence_threshold = cfg.confidence_threshold

y_pred_val = (class_probabilities_val[:, 1] >= confidence_threshold).astype(int)
y_pred_train = (class_probabilites_train[:, 1] >= confidence_threshold).astype(int)

# evaluating model on test set
class_probabilities_test = rf_model.predict_proba(X_test)

y_pred_test = (class_probabilities_test[:, 1] >= confidence_threshold).astype(int)

precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
misclassification_rate_test = 1 - accuracy_score(y_test, y_pred_test)

# saving the model
dump(rf_model, f'../model_files/rf_model_{site}.joblib')
print(f"Model created and saved: models/model_files/rf_model_{site}.joblib")

#-------------------------------------------------------------
# ADDING TO STATS CSV
stats_csv_path = '../model_stats.csv'

new_row = {
    'site': site,
    'model_type': 'RF',
    'train_range': train_range,
    'val_range': val_range,
    'test_range': test_range,
    'train_len': train_len,
    'val_len': val_len,
    'test_len': test_len,
    'precision_test': precision_test,
    'recall_test': recall_test,
    'f1_test': f1_test,
    'misclassification_rate_test': misclassification_rate_test,}

# overwriting the row in the stats CSV if it exists, otherwise appending
if os.path.exists(stats_csv_path):
    stats_df = pd.read_csv(stats_csv_path)
    mask = ~((stats_df['site'] == site) & (stats_df['model_type'] == 'RF'))
    stats_df = stats_df[mask]
    stats_df = pd.concat([stats_df, pd.DataFrame([new_row])], ignore_index=True)
    stats_df.to_csv(stats_csv_path, index=False)

else:
    stats_df = pd.DataFrame([new_row])
    stats_df.to_csv(stats_csv_path, index=False)

#-------------------------------------------------------------
print(f"Model statistics saved: models/model_stats.csv.")
print("")
#-------------------------------------------------------------
# END OF SCRIPT