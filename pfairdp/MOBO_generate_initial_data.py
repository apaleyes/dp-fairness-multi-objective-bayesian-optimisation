import math
import torch

from botorch.utils.sampling import draw_sobol_samples
from utils import append_row_to_csv, get_device

from MOBO_evaluate_pipeline import evaluate_at_params, revert_privacy_value, revert_fairness_value, revert_accuracy_value

tkwargs = {
    "dtype": torch.double,
    "device": get_device(),
}

def generate_initial_data(manual_problem_bounds, n = 6, log_file = None, final_results_path = None):
    # Generate training data by drawing random samples in the search space
    train_x = draw_sobol_samples(
        bounds = manual_problem_bounds, n = 1, q = n, seed = torch.randint(1000000, (1,)).item()
    ).squeeze(0)

    train_obj_true = []

    for i, x in enumerate(train_x.detach().cpu().numpy()):
        num_epochs = x[0]
        noise_multiplier = x[1]
        learning_rate = x[2]
        max_grad_norm = x[3]
        repair_level = x[4]
        batch_size = x[5]

        params = {}
        params['number_of_epochs'] = math.floor(num_epochs)
        params['repair_level'] = repair_level
        params['noise_multiplier'] = noise_multiplier
        params['clipping_norm'] = max_grad_norm
        params['learning_rate'] = learning_rate
        params['batch_size'] = math.floor(batch_size)

        log_file.write('\nRan the pipeline once\n')
        log_file.flush()

        (MOBO_processed_accuracy, MOBO_processed_fairness, MOBO_processed_privacy_budget) = evaluate_at_params(params, log_file)

        train_obj_true.append([MOBO_processed_accuracy, MOBO_processed_fairness, MOBO_processed_privacy_budget])

        append_row_to_csv(final_results_path, [
            str(params['number_of_epochs']),
            str(params['repair_level']),
            str(params['batch_size']),
            str(params['learning_rate']),
            str(params['clipping_norm']),
            str(params['noise_multiplier']),
            str(revert_accuracy_value(MOBO_processed_accuracy)),
            str(revert_privacy_value(MOBO_processed_privacy_budget)),
            str(revert_fairness_value(MOBO_processed_fairness))
        ])

    return train_x.to(**tkwargs), torch.tensor(train_obj_true).to(**tkwargs)
