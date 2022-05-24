from aif360.metrics import BinaryLabelDatasetMetric

def measure_disparate_impact(binary_label_dataset, privileged_groups, unprivileged_groups):
    binary_label_dataset_metric = BinaryLabelDatasetMetric(
        dataset = binary_label_dataset,
        unprivileged_groups = unprivileged_groups,
        privileged_groups = privileged_groups
    )

    return binary_label_dataset_metric.disparate_impact()

def measure_risk_difference(binary_label_dataset, privileged_groups, unprivileged_groups):
    binary_label_dataset_metric = BinaryLabelDatasetMetric(
        dataset = binary_label_dataset,
        unprivileged_groups = unprivileged_groups,
        privileged_groups = privileged_groups
    )

    return abs(binary_label_dataset_metric.statistical_parity_difference())