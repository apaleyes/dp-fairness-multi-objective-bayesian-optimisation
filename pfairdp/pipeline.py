import numpy as np

from fairness_module.fairness_preprocessing import fairness_preprocessing_dir, no_preprocessing
from fairness_module.fairness_postprocessing import reject_option_classification
from fairness_module.fairness_metrics import measure_risk_difference
from training_module.trainer import train, test
from DP_module.enforce_DP import make_private

from utils import to_data_loader, write_to_log_file, get_device, get_data_loader_from_binary_label_dataset

from sklearn.metrics import accuracy_score

def run_pipeline(
            model, 
            optimizer,
            loss_funct, 
            acc_params, fairness_preprocessing_params, fairness_postprocessing_params, dp_params, 
            sensitive_attribute,
            train_binary_label_dataset, test_binary_label_dataset,
            privileged_groups, unprivileged_groups, 
            batch_size,
            log_file,
            device = get_device()):
    
    ###########################
    # Use the fairness module #
    ###########################

    if not bool(fairness_preprocessing_params):
        write_to_log_file(log_file, '\n*** No fairness preprocessing ***\n')
        (X_tr, X_te, y_tr, y_te, preprocessed_train_binary_label_dataset, preprocessed_test_binary_label_dataset) = no_preprocessing(sensitive_attribute, train_binary_label_dataset, test_binary_label_dataset)
    else:
        write_to_log_file(log_file, '\n*** Preprocessing the datasets ***\n')
        (X_tr, X_te, y_tr, y_te, preprocessed_train_binary_label_dataset, preprocessed_test_binary_label_dataset) = fairness_preprocessing_dir(sensitive_attribute, fairness_preprocessing_params, train_binary_label_dataset, test_binary_label_dataset)

    ##############################################################################################################################
    ################################################ Metrics before preprocessing ################################################
    fairness_metric_on_train_before_preprocessing = measure_risk_difference(train_binary_label_dataset, privileged_groups, unprivileged_groups)
    fairness_metric_on_train_after_preprocessing = measure_risk_difference(preprocessed_train_binary_label_dataset, privileged_groups, unprivileged_groups)
    fairness_metric_on_test_before_preprocessing = measure_risk_difference(test_binary_label_dataset, privileged_groups, unprivileged_groups)
    ##############################################################################################################################

    (train_data_loader, test_data_loader) = to_data_loader(X_tr, X_te, y_tr, y_te, batch_size)

    #####################
    # Use the DP module #
    #####################
    (model, optimizer, train_data_loader, privacy_engine) = make_private(model, optimizer, train_data_loader, dp_params)
    
    ###########################
    # Use the training module #
    ###########################
    for epoch in range(1, acc_params['number_of_epochs'] + 1):
        train(model, loss_funct, train_data_loader, optimizer, epoch, verbose = True, device = device, log_file = log_file)
        test_loss_per_epoch, test_acc_per_epoch, _ = test(model, loss_funct, test_data_loader, device = device, log_file = log_file)
        
        write_to_log_file(log_file, 'Epoch {} Test set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(epoch, test_loss_per_epoch, 100. * test_acc_per_epoch))

    # Evaluate the trained model
    _, acc, final_predictions, test_probs = test(model, loss_funct, test_data_loader, device = device, log_file = log_file, return_probs = True)

    # Update the test dataset with the model's predictions in order to measure the final fairness metric
    # and potentially postprocess the dataset
    test_set_after_evaluations = preprocessed_test_binary_label_dataset.copy()
    test_set_after_evaluations.labels = np.reshape(
        np.asarray(final_predictions), 
        newshape = test_set_after_evaluations.labels.shape
        )
    test_set_after_evaluations.scores = np.reshape(
        np.asarray(test_probs),
        newshape = test_set_after_evaluations.scores.shape
    )

    # If postprocessing is enabled
    if bool(fairness_postprocessing_params):
        valid_binary_label_dataset = fairness_postprocessing_params['valid_binary_label_dataset']

        if bool(fairness_preprocessing_params):
            # If preprocessing and postprocessing are combined, we also need to preprocess the validation dataset
            (_, _, _, _, valid_binary_label_dataset, _) = fairness_preprocessing_dir(sensitive_attribute, fairness_preprocessing_params, valid_binary_label_dataset, valid_binary_label_dataset)

        valid_data_loader = get_data_loader_from_binary_label_dataset(valid_binary_label_dataset, sensitive_attribute, batch_size)
        _, _, final_predictions_valid, prediction_probs_valid = test(model, loss_funct, valid_data_loader, device = device, log_file = log_file, return_probs = True)
        valid_binary_label_dataset_after_predictions = valid_binary_label_dataset.copy()
        valid_binary_label_dataset_after_predictions.labels = np.reshape(
            np.asarray(final_predictions_valid), 
            newshape = valid_binary_label_dataset_after_predictions.labels.shape)
        valid_binary_label_dataset_after_predictions.scores = np.reshape(
            np.asarray(prediction_probs_valid), 
            newshape = valid_binary_label_dataset_after_predictions.scores.shape
        )

        default_ROC_ub = 0.05
        default_ROC_lb = -0.05

        # We do not use the default bounds for the second automated experiment
        if 'custom_ub' in fairness_postprocessing_params:
            default_ROC_ub = fairness_postprocessing_params['custom_ub']

        if 'custom_lb' in fairness_postprocessing_params:
            default_ROC_lb = fairness_postprocessing_params['custom_lb']

        test_set_after_evaluations = reject_option_classification(
            privileged_groups = privileged_groups,
            unprivileged_groups = unprivileged_groups,
            train_bl_dataset_true = valid_binary_label_dataset,
            train_bl_dataset_predicted = valid_binary_label_dataset_after_predictions,
            test_bl_dataset_predicted = test_set_after_evaluations,
            custom_ub = default_ROC_ub,
            custom_lb = default_ROC_lb
            )

        # Re-evaluate the accuracy of the model
        acc = accuracy_score(test_binary_label_dataset.labels, test_set_after_evaluations.labels)

    #################################################################################################
    ### Measure fairness on the test set using the model's (optionally postprocessed) predictions ###
    #################################################################################################
    fairness_metric_on_test_after_preprocessing_and_predictions = measure_risk_difference(
        test_set_after_evaluations, 
        privileged_groups = privileged_groups, 
        unprivileged_groups = unprivileged_groups
        )

    epsilon_to_return = 'N/A' # If the privacy module is disabled
    if bool(dp_params):
        epsilon_to_return = privacy_engine.accountant.get_epsilon(delta = 1e-5)

    return (
            acc,
            epsilon_to_return,
            fairness_metric_on_train_before_preprocessing,
            fairness_metric_on_train_after_preprocessing,
            fairness_metric_on_test_before_preprocessing,
            fairness_metric_on_test_after_preprocessing_and_predictions
            )