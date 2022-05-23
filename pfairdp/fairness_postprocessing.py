import numpy as np
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.postprocessing import RejectOptionClassification, EqOddsPostprocessing, CalibratedEqOddsPostprocessing

def reject_option_classification(privileged_groups, unprivileged_groups, 
        train_bl_dataset_true, 
        train_bl_dataset_predicted,
        test_bl_dataset_predicted,
        custom_ub = 0.05,
        custom_lb = -0.05
        ):

    ROC = RejectOptionClassification(
        unprivileged_groups = unprivileged_groups,
        privileged_groups = privileged_groups,
        metric_ub= custom_ub, metric_lb = custom_lb
        )

    ROC = ROC.fit(train_bl_dataset_true, train_bl_dataset_predicted)

    return ROC.predict(test_bl_dataset_predicted)