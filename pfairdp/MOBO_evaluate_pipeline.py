import torch
import math

from utils import get_device, write_to_log_file
from data_load import load_ADULT_from_AIF, load_meps
from pipeline import run_pipeline
from models import SNNMedium, SNNSmall

# Transformations described in the paper in order to make the output domain [-inf, inf] for modelling with GPs
# and turn the optimisation process into maximisation across all objectives (required by BoTorch)
def process_fairness_value(val):
    return math.log(1 - val) - math.log(val)

def revert_fairness_value(val):
    return 1 / (1 + math.e**(val))

def process_privacy_value(val):
    return - math.log(val)

def revert_privacy_value(val):
    return math.e**(-val)

def process_accuracy_value(val):
    return math.log(val) - math.log(1 - val)

def revert_accuracy_value(val):
    return 1 / (1 + math.e**(-val))

use_meps = False

if use_meps:
    (train_binary_label_dataset, test_binary_label_dataset) = load_meps()
    sensitive_attribute = 'RACE'
    privileged_groups = [{'RACE': 1.0}]
    unprivileged_groups = [{'RACE': 0.0}]
    target_attribute = 'UTILIZATION'
else:
    (train_binary_label_dataset, test_binary_label_dataset) = load_ADULT_from_AIF(use_all_features = False, should_scale = True)
    sensitive_attribute = 'sex'
    privileged_groups = [{'sex': 1.0}]
    unprivileged_groups = [{'sex': 0.0}]
    target_attribute = 'income-per-year'

device = get_device()

def evaluate_at_params(params, log_file):
    write_to_log_file(log_file, "Evaluation starting\n")
    acc_params = {}
    acc_params['number_of_epochs'] = params['number_of_epochs']
    acc_params['batch_size'] = params['batch_size']
    acc_params['learning_rate'] = params['learning_rate']

    fairness_preprocessing_params = {}
    fairness_preprocessing_params['repair_level'] = params['repair_level']

    fairness_postprocessing_params = {}

    dp_params = {}
    dp_params['noise_multiplier'] = params['noise_multiplier']
    dp_params['max_grad_norm'] = params['clipping_norm']
    dp_params['delta'] = 10**(-5)

    # Instantiate the model
    num_features_in_adult = len(train_binary_label_dataset.convert_to_dataframe()[0].drop([target_attribute], axis = 1).columns) - 1

    if use_meps:
        model = SNNMedium(num_features_in_adult).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr = acc_params['learning_rate'])
    else:
        model = SNNSmall(num_features_in_adult).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = acc_params['learning_rate'])
        
    model.train()


    (acc, 
    privacy_budget, 
    _, _, _, 
    fairness_metric_after_test) = run_pipeline(model, 
                                            optimizer,
                                            torch.nn.functional.binary_cross_entropy, 
                                            acc_params, 
                                            fairness_preprocessing_params, fairness_postprocessing_params, 
                                            dp_params,
                                            sensitive_attribute,
                                            train_binary_label_dataset, test_binary_label_dataset,
                                            privileged_groups, unprivileged_groups,
                                            batch_size = acc_params['batch_size'],
                                            log_file =  log_file
                                            )

    write_to_log_file(log_file, "Evaluation completed\n")
    write_to_log_file(log_file, f'\nAccuracy {acc} | Fairness {fairness_metric_after_test} | Privacy budget {privacy_budget}\n')
    write_to_log_file(log_file, f'Accuracy {acc} | Fairness {fairness_metric_after_test} | Privacy budget {privacy_budget}\n')
    write_to_log_file(log_file, f'Accuracy {acc} | Fairness {fairness_metric_after_test} | Privacy budget {privacy_budget}\n')
    write_to_log_file(log_file, '\n\n')

    # We add a small error to these results so that the inverse of the MOBO transformations does not fail
    if fairness_metric_after_test == 0:
        fairness_metric_after_test = 0.0001
    
    if fairness_metric_after_test == 1:
        fairness_metric_after_test = 1 - 0.0001

    return (
        process_accuracy_value(acc), 
        process_fairness_value(fairness_metric_after_test), 
        process_privacy_value(privacy_budget)
        )