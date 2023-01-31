from aif360.algorithms.postprocessing import RejectOptionClassification

def reject_option_classification(privileged_groups, unprivileged_groups, 
        train_bl_dataset_true, 
        train_bl_dataset_predicted,
        test_bl_dataset_predicted,
        custom_ub = 0.05,   # This is the default upper bound used by ROC
        custom_lb = -0.05   # This is the default lower bound used by ROC
        ):

    ROC = RejectOptionClassification(
        unprivileged_groups = unprivileged_groups,
        privileged_groups = privileged_groups,
        metric_ub= custom_ub, metric_lb = custom_lb
        )

    ROC = ROC.fit(train_bl_dataset_true, train_bl_dataset_predicted)

    return ROC.predict(test_bl_dataset_predicted)