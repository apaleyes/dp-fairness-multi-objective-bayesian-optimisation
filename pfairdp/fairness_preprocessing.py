import numpy as np

from aif360.algorithms.preprocessing import DisparateImpactRemover, LFR, OptimPreproc, Reweighing
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools

def no_preprocessing(sensitive_attribute, train_bl_dataset, test_bl_dataset):
    X_tr = np.delete(train_bl_dataset.features, train_bl_dataset.feature_names.index(sensitive_attribute), axis=1)
    X_te = np.delete(test_bl_dataset.features, test_bl_dataset.feature_names.index(sensitive_attribute), axis=1)

    y_tr = train_bl_dataset.labels.ravel()
    y_te = test_bl_dataset.labels.ravel()

    return (X_tr, X_te, y_tr, y_te, train_bl_dataset, test_bl_dataset)

def fairness_preprocessing_dir(sensitive_atrribute, fairness_params, train_bl_dataset, test_bl_dataset):
    if not 'repair_level' in fairness_params:
        raise Exception('A repair level must be provided.')

    index = train_bl_dataset.feature_names.index(sensitive_atrribute)

    dir = DisparateImpactRemover(repair_level = fairness_params['repair_level'], sensitive_attribute = sensitive_atrribute)

    preprocessed_train_binary_label_dataset = dir.fit_transform(train_bl_dataset)
    preprocessed_test_binary_label_dataset = dir.fit_transform(test_bl_dataset)

    X_tr = np.delete(preprocessed_train_binary_label_dataset.features, index, axis=1)
    X_te = np.delete(preprocessed_test_binary_label_dataset.features, index, axis=1)
    
    y_tr = preprocessed_train_binary_label_dataset.labels.ravel()
    y_te = test_bl_dataset.labels.ravel()

    return (X_tr, X_te, y_tr, y_te, preprocessed_train_binary_label_dataset, preprocessed_test_binary_label_dataset)

def fairness_preprocessing_LFR(sensitive_atrribute, unprivileged_groups, privileged_groups, fairness_params, train_bl_dataset, test_bl_dataset):
    lfr = LFR(
        unprivileged_groups = unprivileged_groups, privileged_groups = privileged_groups,
        k = fairness_params['k'],
        Ax = fairness_params['Ax'],
        Ay = fairness_params['Ay'],
        Az = fairness_params['Az'] 
    )

    index = train_bl_dataset.feature_names.index(sensitive_atrribute)

    preprocessed_train_binary_label_dataset = lfr.transform(train_bl_dataset)
    preprocessed_test_binary_label_dataset = lfr.transform(test_bl_dataset)

    X_tr = np.delete(preprocessed_train_binary_label_dataset.features, index, axis=1)
    X_te = np.delete(preprocessed_test_binary_label_dataset.features, index, axis=1)
    
    y_tr = preprocessed_train_binary_label_dataset.labels.ravel()
    y_te = test_bl_dataset.labels.ravel()

    return (X_tr, X_te, y_tr, y_te, preprocessed_train_binary_label_dataset, preprocessed_test_binary_label_dataset)

def fairness_preprocessing_optim(sensitive_atrribute, unprivileged_groups, privileged_groups, fairness_params, train_bl_dataset, test_bl_dataset):
    optim_options = {
        "distortion_fun": fairness_params['distortion_fun'],
        "epsilon": fairness_params['epsilon'],
        "clist": fairness_params['clist'],
        "dlist": fairness_params['dlist']
    }

    OP = OptimPreproc(OptTools, optim_options,
                  unprivileged_groups = unprivileged_groups,
                  privileged_groups = privileged_groups)

    OP = OP.fit(train_bl_dataset)

    index = train_bl_dataset.feature_names.index(sensitive_atrribute)

    preprocessed_train_binary_label_dataset = OP.transform(train_bl_dataset, transform_Y=True)
    preprocessed_train_binary_label_dataset = train_bl_dataset.align(preprocessed_train_binary_label_dataset)

    preprocessed_test_binary_label_dataset = OP.transform(test_bl_dataset, transform_Y=True)
    preprocessed_test_binary_label_dataset = test_bl_dataset.align(preprocessed_test_binary_label_dataset)

    X_tr = np.delete(preprocessed_train_binary_label_dataset.features, index, axis=1)
    X_te = np.delete(preprocessed_test_binary_label_dataset.features, index, axis=1)
    
    y_tr = preprocessed_train_binary_label_dataset.labels.ravel()
    y_te = test_bl_dataset.labels.ravel()

    return (X_tr, X_te, y_tr, y_te, preprocessed_train_binary_label_dataset, preprocessed_test_binary_label_dataset)

def fairness_preprocessing_reweighing(sensitive_atrribute, unprivileged_groups, privileged_groups, fairness_params, train_bl_dataset, test_bl_dataset):
    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    RW.fit(train_bl_dataset)

    index = train_bl_dataset.feature_names.index(sensitive_atrribute)

    preprocessed_train_binary_label_dataset = RW.transform(train_bl_dataset)
    preprocessed_test_binary_label_dataset = RW.transform(test_bl_dataset)

    X_tr = np.delete(preprocessed_train_binary_label_dataset.features, index, axis=1)
    X_te = np.delete(preprocessed_test_binary_label_dataset.features, index, axis=1)
    
    y_tr = preprocessed_train_binary_label_dataset.labels.ravel()
    y_te = test_bl_dataset.labels.ravel()

    return (X_tr, X_te, y_tr, y_te, preprocessed_train_binary_label_dataset, preprocessed_test_binary_label_dataset)
