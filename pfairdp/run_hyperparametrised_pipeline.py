# Common imports
import numpy as np
import os
import torch
from datetime import datetime

# Imports from workspace
from hyperparameter_mesh import generate_grid_hyperparameters_instances, generate_random_hyperparameter_instances
from utils import append_row_to_csv, get_device, convert_acc_for_csv, log_configuration, write_dataset_to_file
from data_load import load_ADULT_from_AIF, load_meps
from pipeline import run_pipeline
from models import SNNSmall, SNNMedium

def main():
    device = get_device()

    # Create files for storing results
    current_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    experiment_dir = "dp-fairness-multi-objective-bayesian-optimisation/final_experiments_Pareto/main_experiments_" + current_time + '/'

    log_file_path = experiment_dir + 'config2_log.txt'
    final_results_path = experiment_dir + 'results_config2_ADULT.csv'

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir, exist_ok = True)

    # Open file for storing results
    log_file = open(log_file_path, "a")

    # Parameters and metrics we want to store for each pipeline run
    append_row_to_csv(final_results_path, [
        'Number of epochs', 
        'Repair level', 
        'Batch size', 
        'Learning rate',
        'Clipping Norm', 
        'Noise multiplier', 
        'Classification error', 
        'Privacy budget', 
        'Fairness before', 
        'Fairness after'
        ])

    ####################################################################
    ### Load the dataset and define privileged / unprivileged groups ###
    ####################################################################

    should_load_meps = False

    if should_load_meps:
        (train_binary_label_dataset, test_binary_label_dataset) = load_meps()
        write_dataset_to_file(train_binary_label_dataset.convert_to_dataframe()[0], experiment_dir + f'meps.csv')
        sensitive_attribute = 'RACE'
        privileged_groups = [{'RACE': 1.0}]
        unprivileged_groups = [{'RACE': 0.0}]
        target_attribute = 'UTILIZATION'
    else:
        (train_binary_label_dataset, test_binary_label_dataset) = load_ADULT_from_AIF(should_scale = True, use_all_features = False)
        sensitive_attribute = 'sex'
        privileged_groups = [{'sex': 1.0}]
        unprivileged_groups = [{'sex': 0.0}]
        target_attribute = 'income-per-year'

    ## Sampling params

    run_random_search = True
    run_grid_search = False

    if run_random_search:
        num_instances = 300

        hyperparam_ranges = {}
        hyperparam_ranges['number_of_epochs'] = {'type': 'uniform' , 'params': [30, 128], 'round_to_int': True}
        hyperparam_ranges['batch_size'] = {'type': 'uniform' , 'params': [16, 64], 'round_to_int': True}
        hyperparam_ranges['learning_rate'] = {'type': 'exponential', 'params': [10, 1e-3], 'reject_greater_than': 1e-1}

        hyperparam_ranges['noise_multiplier'] = {'type': 'uniform' , 'params': [1, 5]}
        hyperparam_ranges['clipping_norm'] = {'type': 'uniform' , 'params': [0.1, 2]}
        
        hyperparam_ranges['repair_level'] = {'type': 'uniform' , 'params': [0, 1]}

        search_space = generate_random_hyperparameter_instances(hyperparam_ranges, num_instances = num_instances, num_replications = 1)
    elif run_grid_search:
        points_per_param = 4

        hyperparam_ranges = {}
        hyperparam_ranges['number_of_epochs'] = {'min': 30, 'max': 128, 'round_to_int': True}
        hyperparam_ranges['batch_size'] = {'value': 32, 'round_to_int': True}
        hyperparam_ranges['learning_rate'] = {'value': 0.01}

        hyperparam_ranges['noise_multiplier'] = {'min': 1, 'max': 5}
        hyperparam_ranges['clipping_norm'] = {'min': 0.1 , 'max': 2}

        hyperparam_ranges['repair_level'] = {'min': 0 , 'max': 1}

        search_space = generate_grid_hyperparameters_instances(hyperparam_ranges, points_per_param = points_per_param, num_replications = 1)

    for params_iteration in search_space:

        acc_params = {}
        acc_params['number_of_epochs'] = params_iteration['number_of_epochs']
        acc_params['batch_size'] = params_iteration['batch_size']

        fairness_preproc_params = {}
        fairness_preproc_params['repair_level'] = params_iteration['repair_level']

        fairness_postproc_params = {}

        dp_params = {}
        dp_params['noise_multiplier'] = params_iteration['noise_multiplier']
        dp_params['max_grad_norm'] = params_iteration['clipping_norm']

        # Instantiate the model
        num_features_in_adult = len(train_binary_label_dataset.convert_to_dataframe()[0].drop([target_attribute], axis = 1).columns) - 1
        if should_load_meps:
            model = SNNMedium(num_features_in_adult).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr = params_iteration['learning_rate'])
        else:
            model = SNNSmall(num_features_in_adult).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr = params_iteration['learning_rate'])
        model.train()

        (acc, privacy_budget, 
        fairness_metric_before_train, fairness_metric_after_train, 
        fairness_metric_before_test, fairness_metric_after_test) = run_pipeline(model, 
                                                                                optimizer,
                                                                                torch.nn.functional.binary_cross_entropy, 
                                                                                acc_params, fairness_preproc_params, fairness_postproc_params, dp_params,
                                                                                sensitive_attribute,
                                                                                train_binary_label_dataset, test_binary_label_dataset,
                                                                                privileged_groups, unprivileged_groups,
                                                                                params_iteration['batch_size'],
                                                                                log_file
                                                                                )

        log_configuration(log_file, 
            acc, 
            privacy_budget, 
            fairness_metric_before_train, fairness_metric_after_train, 
            fairness_metric_before_test, fairness_metric_after_test
            )

        append_row_to_csv(final_results_path, [
            str(params_iteration['number_of_epochs']),
            str(params_iteration['repair_level']),
            str(params_iteration['batch_size']),
            str(params_iteration['learning_rate']),
            str(params_iteration['clipping_norm']),
            str(params_iteration['noise_multiplier']),
            convert_acc_for_csv(acc),
            str(privacy_budget),
            str(fairness_metric_before_test), 
            str(fairness_metric_after_test)
        ])

if __name__ == "__main__":
    main()