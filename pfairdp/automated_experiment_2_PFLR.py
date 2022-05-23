# Common imports
import os
import torch
from datetime import datetime

# Imports from workspace
from utils import append_row_to_csv, get_device, convert_acc_for_csv, log_configuration
from data_load import load_ADULT_from_AIF
from pipeline import run_pipeline
from models import EquivalentLogisticRegression

device = get_device()

# Create files for storing results
current_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
experiment_dir = "/home/bf319/dp-fairness-multi-objective-bayesian-optimisation/final_experiments_Pareto/automated_experiment_2_" + current_time + '/PFLR/'

log_file_path = experiment_dir + 'log.txt'
final_results_path = experiment_dir + 'results.csv'

if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir, exist_ok = True)

# Open file for storing results
log_file = open(log_file_path, "a")

def main():
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

    (train_binary_label_dataset, test_binary_label_dataset) = load_ADULT_from_AIF(should_scale = True, use_all_features = False, validation = False)

    sensitive_attribute = 'sex'
    privileged_groups = [{'sex': 1.0}]
    unprivileged_groups = [{'sex': 0.0}]
    target_attribute = 'income-per-year'

    for noise_multiplier in [7, 1.07, 0.48]:
        for _ in range(20):
            params_iteration = {}

            params_iteration['learning_rate'] = 1e-3

            acc_params = {}
            acc_params['number_of_epochs'] = 100
            acc_params['batch_size'] = 20

            fairness_preprocessing_params = {}
            fairness_preprocessing_params['repair_level'] = 1 # Comment this to run PrivLR

            fairness_postprocessing_params = {}

            '''
            3.3 => Epsilon = 0.1 (for delta = 1e-5)
            1 => Epsilon = 0.99 (for delta = 1e-5)
            0.567 => Epsilon = 10.3 (for delta = 1e-5)
            0.26 => Epsilon = 100 (for delta = 1e-5)
            '''
            dp_params = {}
            dp_params['noise_multiplier'] = noise_multiplier
            dp_params['max_grad_norm'] = 2

            # Instantiate the model
            num_features_in_adult = len(train_binary_label_dataset.convert_to_dataframe()[0].drop([target_attribute], axis = 1).columns) - 1
            model = EquivalentLogisticRegression(num_features_in_adult).to(device)
            model.train()

            optimizer = torch.optim.Adam(model.parameters(), lr = params_iteration['learning_rate'])

            (acc, 
            privacy_budget, 
            fairness_metric_before_train, fairness_metric_after_train, 
            fairness_metric_before_test, fairness_metric_after_test) = run_pipeline(
                                                                                    model, 
                                                                                    optimizer,
                                                                                    torch.nn.BCELoss(), 
                                                                                    acc_params, 
                                                                                    fairness_preprocessing_params, fairness_postprocessing_params,
                                                                                    dp_params,
                                                                                    sensitive_attribute,
                                                                                    train_binary_label_dataset, test_binary_label_dataset,
                                                                                    privileged_groups, unprivileged_groups,
                                                                                    batch_size = acc_params['batch_size'],
                                                                                    log_file = log_file,
                                                                                    )

            log_configuration(log_file, 
                acc, 
                privacy_budget, 
                fairness_metric_before_train, fairness_metric_after_train, 
                fairness_metric_before_test, fairness_metric_after_test
                )

            if 'repair_level' in fairness_preprocessing_params: # PFLR
                append_row_to_csv(final_results_path, [
                    str(acc_params['number_of_epochs']),
                    str(fairness_preprocessing_params['repair_level']),
                    str(acc_params['batch_size']),
                    str(params_iteration['learning_rate']),
                    str(dp_params['max_grad_norm']),
                    str(dp_params['noise_multiplier']),
                    convert_acc_for_csv(acc),
                    str(privacy_budget),
                    str(fairness_metric_before_test), 
                    str(fairness_metric_after_test)
                ])
            else:
                append_row_to_csv(final_results_path, [ # PrivLR
                    str(acc_params['number_of_epochs']),
                    'N/A',
                    str(acc_params['batch_size']),
                    str(params_iteration['learning_rate']),
                    str(dp_params['max_grad_norm']),
                    str(dp_params['noise_multiplier']),
                    convert_acc_for_csv(acc),
                    str(privacy_budget),
                    str(fairness_metric_before_test), 
                    str(fairness_metric_after_test)
                ])

if __name__ == "__main__":
    main()