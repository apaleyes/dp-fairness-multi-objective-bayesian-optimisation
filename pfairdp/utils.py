import csv
import torch
import torch.utils.data as data_utils

def to_data_loader(X_tr, X_te, y_tr, y_te, batch_size):
    train_data_loader = data_utils.DataLoader(data_utils.TensorDataset(torch.Tensor(X_tr), torch.Tensor(y_tr)), batch_size = batch_size)
    test_data_loader = data_utils.DataLoader(data_utils.TensorDataset(torch.Tensor(X_te), torch.Tensor(y_te)), batch_size = batch_size)

    return (train_data_loader, test_data_loader)

def write_dataset_to_file(df, full_path):
    df.to_csv(full_path)

def append_row_to_csv(file_path, values):
    with open(file_path, 'a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter = ',')
        csv_writer.writerow(values)

        csvfile.flush()

def get_device():
    if torch.cuda.is_available():  
        dev = "cuda" 
    else:  
        dev = "cpu"  

    device = torch.device(dev)

    return device

def truncate_df(adult_data_train, adult_data_test, num_records_to_keep = 0):
    adult_data_train = adult_data_train.head(num_records_to_keep)
    adult_data_test = adult_data_test.head(num_records_to_keep)

    return (adult_data_train, adult_data_test)

def convert_acc_for_csv(acc):
    try:
        val = 1 - float(acc)
        return val
    except ValueError:
        return 'N/A'

import math

def write_to_log_file(log_file_handle, line):
    log_file_handle.write(line)
    log_file_handle.flush()

def log_exception(log_file_handle, e):
    log_file_handle.write('******************* \n')
    log_file_handle.write('Some error happened \n')
    log_file_handle.write('******************* \n')
    log_file_handle.write(str(e))
    log_file_handle.write('\n\n\n')
    log_file_handle.flush()

def log_configuration(log_file_handle, acc, privacy_budget, fairness_metric_before_train, fairness_metric_after_train, fairness_metric_before_test, fairness_metric_after_test):
    log_file_handle.write(f'\nFinal accuracy on the test set {acc} \n')
    log_file_handle.write(f'Privacy budget: {privacy_budget} \n')
    log_file_handle.write(f'(Train) Fairness metric before AIF preprocessing: {fairness_metric_before_train} \n')
    log_file_handle.write(f'(Train) Fairness metric after AIF preprocessing: {fairness_metric_after_train} \n')
    log_file_handle.write(f'(Test) Fairness metric before AIF preprocessing: {fairness_metric_before_test} \n')
    log_file_handle.write(f'(Test) Fairness metric after AIF preprocessing + predictions: {fairness_metric_after_test} \n')
    log_file_handle.write('\n\n')
    log_file_handle.flush()

import numpy as np

def get_data_loader_from_binary_label_dataset(binary_label_dataset, sensitive_attribute, batch_size):
    X = np.delete(binary_label_dataset.features, binary_label_dataset.feature_names.index(sensitive_attribute), axis=1)
    y = binary_label_dataset.labels.ravel()

    return data_utils.DataLoader(data_utils.TensorDataset(torch.Tensor(X),torch.Tensor(y)), batch_size = batch_size)