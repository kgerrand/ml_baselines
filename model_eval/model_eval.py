'''
Code to Evaluate Model Performance and Classify Anomalies

This script is used to evaluate the performance of machine learning models for a given site and compound. It loads the model, processes the data, and generates predictions. The results are then saved for further analysis.
It loads the model, processes the data, and generates predictions. The results are then saved for further analysis in a csv file, one per site.

'''

#-------------------------------------------------------------
import pandas as pd
import sys, os, argparse
sys.path.append('../')
import functions as f
import config as cfg

data_path = cfg.path_root/'data_files'

#-------------------------------------------------------------
def model_eval(site, model_type):
    """
    Evaluate the ML model for a given site and model type.
    Saves results to a CSV file in the model_results directory.
    
    Parameters:
    site (str): The AGAGE site to evaluate.
    model_type (str): The type of model to evaluate ('nn' or 'rf').
    
    Returns:
    None
    """
    # Load the model
    model_name = f'{model_type}_model_{site}.joblib'
    model = f.access_model(model_name)

    # Determine the compounds available for the site
    print(f"Checking available compounds...")
    compound_list = ['CH4', 'CF4', 'CFC-12', 'CH2Cl2', 'CH3Br',
                     'HCFC-22', 'HFC-125', 'HFC-134A', 'N2O', 'SF6']
    site_compounds = []
    for compound in compound_list:
        data_file = data_path / 'saved_files' / f'data_ds_{compound.lower()}_{site}.nc'
        if data_file.exists():
            site_compounds.append(compound)
    print(f"{len(site_compounds)} compounds available for {site}.")
    if not site_compounds:
        print(f"No compounds available for {site}. Exiting.")
        return
    print('')    
    
    # Make predictions
    num_processed = 0
    num_site_compounds = len(site_compounds)
    for compound in site_compounds:
        print(f"--- Processing {compound} ({num_processed+1}/{num_site_compounds}) ---")
        results = f.make_predictions(model, site, compound.lower())
        print(f"Predictions made.")
        print('')

        # quantitative analysis
        cv = f.quantify_noise(results, compound.lower())
        print(f"Coefficient of Variation, CV (true baselines): {cv:.3f}")
        mae, rmse, mape = f.calc_statistics(results)
        print(f"MAE={mae:.3f}ppt, RMSE={rmse:.3f}ppt, MAPE={mape:.3f}%")
        print('')

        # quantitative analysis
        f.plot_predictions(results, site, compound.lower(), model_type)
        num_anomalies, anomalies, num_signif_anomalies, signif_anomalies = f.plot_predictions_monthly(results, site, compound.lower(), model_type)
        if num_anomalies > 0:
            print(f"Number of anomalous months: {num_anomalies}")
            print(f"Number of significant anomalies (>10σ): {num_signif_anomalies}/{num_anomalies}")
        else:
            print("No anomalous months detected.")

        # Save results to CSV
        new_row = {
            'compound': compound,
            'coefficient_of_variation': cv,
            'MAE / ppt': mae,
            'RMSE / ppt': rmse,
            'MAPE / %': mape,
            'Num anomalous months': num_anomalies,
            'Anomaly list': anomalies,
            'Num significant anomalies (>10std)': num_signif_anomalies,
            'Significant anomaly list (>10std)': signif_anomalies
        }

        results_csv = os.path.join(f'model_results/{site}', f'{site}_{model_type}.csv')
        if not os.path.exists(f'model_results/{site}'):
            os.makedirs(f'model_results/{site}')
        if os.path.exists(results_csv):
            results_df = pd.read_csv(results_csv)
            mask = ~((results_df['compound'] == compound))
            results_df = results_df[mask]
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            results_df.to_csv(results_csv, index=False)
        else:
            results_df = pd.DataFrame([new_row])
            results_df.to_csv(results_csv, index=False)

        num_processed += 1
        print('\n')

    print('All compounds processed.')
    print(f'Results saved to: model_results/{site}/{site}_{model_type}.csv')


#-------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model evaluation script.")
    parser.add_argument('--site', type=str, required=True, help='Input chosen AGAGE site (e.g., MHD, RPB, CGO, etc.).')
    parser.add_argument('--model_type', type=str, required=True, help='Input chosen model type (NN or RF).')

    args = parser.parse_args()
    site = args.site
    model_type = args.model_type.lower()

    # Validate inputs
    assert site in ['MHD', 'RPB', 'CGO', 'GSN', 'JFJ', 'CMN', 'THD', 'ZEP', 'SMO'], f"Site {site} not recognised."
    assert model_type in ['nn', 'rf'], f"Model type {model_type} not recognised."

    if model_type == 'nn':
        model_name = 'neural network'
    elif model_type == 'rf':
        model_name = 'random forest'

    print(f"Evaluating the \033[1m{site}\033[0;0m \033[1m{model_name}\033[0;0m model...")
    print('')
    model_eval(site, model_type)

    print(f"Model evaluation for {site} completed.")

#-------------------------------------------------------------
