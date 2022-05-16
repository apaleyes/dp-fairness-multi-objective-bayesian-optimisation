# Common imports
import os
import torch
from datetime import datetime

# Imports from workspace
from utils import append_row_to_csv, get_device, convert_acc_for_csv, log_configuration
from data_load import preprocess_adult_paper_based
from pipeline import run_pipeline
from models import SNNSmall

# Create files for storing results
current_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
experiment_dir = "/home/bf319/dp-fairness-multi-objective-bayesian-optimisation/final_experiments_Pareto/automated_experiment_1_" + current_time + '/'

log_file_path = experiment_dir + 'log.txt'
final_results_path = experiment_dir + 'results.csv'

if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir, exist_ok = True)

# Open file for storing results
log_file = open(log_file_path, "a")

def main():
    device = get_device()

    ## Learning params
    batch_size = 20

    # Parameters and metrics we want to store for each pipeline run
    append_row_to_csv(final_results_path, [
        'Noise Multiplier', 
        'Delta', 
        'Classification error', 
        'Privacy budget', 
        'Fairness before (Risk difference)', 
        'Fairness after (Risk difference)'
        ])

    ####################################################################
    ### Load the dataset and define privileged / unprivileged groups ###
    ####################################################################

    (train_binary_label_dataset, valid_binary_label_dataset, test_binary_label_dataset) = preprocess_adult_paper_based()

    sensitive_attribute = 'sex'
    privileged_groups = [{'sex': 1.0}]
    unprivileged_groups = [{'sex': 0.0}]
    target_attribute = 'income-per-year'

    for i in range(20):
        noise_multiplier = 4.03    # Determined using Opacus script for our specific sampling rate and target epsilon
        delta = 0.00001

        params_iteration = {}
        params_iteration['number_of_epochs'] = 20
        params_iteration['clipping_norm'] = 2.0

        acc_params = {}
        acc_params['number_of_epochs'] = params_iteration['number_of_epochs']

        fairness_preprocessing_params = {}

        fairness_postprocessing_params = {
            'valid_binary_label_dataset': valid_binary_label_dataset
        }

        dp_params = {}
        dp_params['noise_multiplier'] = noise_multiplier
        dp_params['max_grad_norm'] = params_iteration['clipping_norm']
        
        # Instantiate the model
        num_features_in_adult = len(train_binary_label_dataset.convert_to_dataframe()[0].drop([target_attribute], axis = 1).columns) - 1
        model = SNNSmall(num_features_in_adult).to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters())

        (acc, 
        privacy_budget, 
        _, _, 
        fairness_metric_before_test, fairness_metric_after_test) = run_pipeline(model = model, 
                                                                                optimizer = optimizer,
                                                                                loss_funct = torch.nn.BCELoss(), 
                                                                                acc_params = acc_params, 
                                                                                fairness_preprocessing_params = fairness_preprocessing_params, 
                                                                                fairness_postprocessing_params = fairness_postprocessing_params,
                                                                                dp_params = dp_params,
                                                                                sensitive_attribute = sensitive_attribute,
                                                                                train_binary_label_dataset = train_binary_label_dataset, 
                                                                                test_binary_label_dataset = test_binary_label_dataset,
                                                                                privileged_groups = privileged_groups, unprivileged_groups = unprivileged_groups,
                                                                                batch_size = batch_size,
                                                                                log_file = log_file,
                                                                                )

        log_configuration(log_file, 
            acc, 
            privacy_budget, 
            'N/A', 'N/A', 
            fairness_metric_before_test, fairness_metric_after_test
            )

        append_row_to_csv(final_results_path, [
            str(noise_multiplier),
            str(delta),
            convert_acc_for_csv(acc),
            str(privacy_budget),
            str(fairness_metric_before_test), 
            str(fairness_metric_after_test)
        ])

if __name__ == "__main__":
    main()