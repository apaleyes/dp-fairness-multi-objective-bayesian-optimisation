import os
import math
import time
import torch
import warnings
import gpytorch

from botorch import fit_gpytorch_model
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import  FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from datetime import datetime
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from utils import append_row_to_csv, log_exception, get_device, write_to_log_file

from MOBO_evaluate_pipeline import evaluate_at_params, revert_privacy_value, revert_fairness_value, revert_accuracy_value, process_fairness_value, process_privacy_value, process_accuracy_value
from MOBO_generate_initial_data import generate_initial_data

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

tkwargs = {
    "dtype": torch.double,
    "device": get_device(),
}

current_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
experiment_dir = "dp-fairness-multi-objective-bayesian-optimisation/final_experiments_Pareto/MOBO_" + current_time + '/'

final_results_path = experiment_dir + 'results.csv'

BATCH_SIZE = 1
NUM_RESTARTS = 20
RAW_SAMPLES = 1024
N_TRIALS = 1        
N_BATCH = 250
MC_SAMPLES = 128    

verbose = True

# This function is copied from https://github.com/amzn/differential-privacy-bayesian-optimization/blob/master/dpareto/hypervolume_improvement/harness.py 
def _train_torch_gp(train_x, train_y, gp_factory):
    model = gp_factory(train_x, train_y)
    model.train()
    model.likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr = 0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    training_iter = 2
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y.reshape(-1))
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (
            i + 1, training_iter, loss.item()
        ))
        optimizer.step()

    return model

# Most of the remaining code in this file is adapted from the BoTorch tutorial on 
# multi-objective Bayesian optimisation https://botorch.org/tutorials/multi_objective_bo
def initialize_model(train_x, train_obj):
    ''' 
        Define models for objective and constraint
        Models are used as a surrogate function for the actual underlying black box function to be optimized.
        In BoTorch, a Model maps a set of design points to a posterior probability distribution of its output(s) over the design points. 
    '''
    models = []

    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i:i+1]

        objectiveGP = _train_torch_gp(train_x, train_y, lambda x, y: SingleTaskGP(x, y))
        models.append(objectiveGP)

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def optimize_qehvi_and_get_observation(model, train_x, sampler, manual_problem_bounds, manual_problem_ref_point, log_file):
    """
        Optimizes the qEHVI (parallel Expected Hypervolume Improvement) 
        acquisition function, and returns a new candidate and observation.
    """
    # Partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        pred = model.posterior(train_x).mean

    partitioning = FastNondominatedPartitioning(
        ref_point = manual_problem_ref_point, 
        Y = pred,
    )

    acq_func = qExpectedHypervolumeImprovement(
        model = model,
        ref_point = manual_problem_ref_point,
        partitioning = partitioning,
        sampler = sampler,
    )

    candidates, _ = optimize_acqf(
        acq_function = acq_func,
        bounds = manual_problem_bounds,
        q = BATCH_SIZE,
        num_restarts = NUM_RESTARTS,
        raw_samples = RAW_SAMPLES,  # used for intialization heuristic
        options = {"batch_limit": 5, "maxiter": 200},
        sequential = True,
    )

    # Observe new values 
    new_x = candidates.detach() 

    new_obj_true = []
    for i, x in enumerate(new_x.cpu().numpy()):
        write_to_log_file(log_file, f"Starting iteration {i} \n\n")
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

        write_to_log_file(log_file, f'Running parameters \n {params} \n\n')

        (acc, fairness, privacy_budget) = evaluate_at_params(params, log_file)

        new_obj_true.append([acc, fairness, privacy_budget])

        append_row_to_csv(final_results_path, [
            str(params['number_of_epochs']),
            str(params['repair_level']),
            str(params['batch_size']),
            str(params['learning_rate']),
            str(params['clipping_norm']),
            str(params['noise_multiplier']),
            str(revert_accuracy_value(acc)),
            str(revert_privacy_value(privacy_budget)),
            "0",
            str(revert_fairness_value(fairness))
        ])
    
    return new_x.to(**tkwargs), torch.tensor(new_obj_true).to(**tkwargs)

def main():
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir, exist_ok = True)
    log_file = open(experiment_dir + 'log.txt', "a")

    write_to_log_file(log_file, f"Device: {tkwargs['device']}\n\n")

    append_row_to_csv(final_results_path, [
        'Number of epochs', 
        'Repair level', 
        'Batch size', 
        'Learning rate',
        'Clipping Norm', 
        'Noise multiplier', 
        'Accuracy', 
        'Privacy budget',
        'Fairness before',
        'Fairness after'
        ])

    try:
        # BoTorch assumes a maximization of all objectives (https://botorch.org/tutorials/multi_objective_bo)
        manual_problem_bounds = torch.FloatTensor([
                                            [30, 1.0, 10**(-3), 0.1, 0, 16],  # Lower bounds 
                                            [128.0, 5.0, 10**(-1), 2, 1, 64]   # Upper bounds
                                            ]).to(**tkwargs)
        '''
            qEHVI requires specifying a reference point, which is the lower bound on the objectives used for computing hypervolume.
            In practice the reference point can be set 
                1)  using domain knowledge to be slightly worse than the lower bound of objective values, 
                    where the lower bound is the minimum acceptable value of interest for each objective
                2)  using a dynamic reference point selection strategy.
        '''
        anti_ideal_point = torch.FloatTensor([
            process_accuracy_value(0.0001),
            process_fairness_value(1 - 0.0001),
            process_privacy_value(1)
            ]).to(**tkwargs)

        # Average over multiple trials
        hvs_qehvi = []
        
        train_x_qehvi, train_obj_true_qehvi = generate_initial_data(manual_problem_bounds, n = 16, log_file = log_file, final_results_path = final_results_path)

        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_true_qehvi)
        
        # Compute hypervolume
        bd = DominatedPartitioning(ref_point = anti_ideal_point, Y = train_obj_true_qehvi)
        volume = bd.compute_hypervolume().item()
        
        hvs_qehvi.append(volume)
        
        iteration = 0
        while iteration < N_BATCH:
            t0 = time.time()
            
            # Fit the models
            fit_gpytorch_model(mll_qehvi)
            
            # Define the qEHVI acquisition module using a QMC sampler
            qehvi_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
            
            # Optimize acquisition functions and get new observations
            new_x_qehvi, new_obj_true_qehvi = optimize_qehvi_and_get_observation(
                model = model_qehvi, 
                train_x = train_x_qehvi, 
                sampler = qehvi_sampler,
                manual_problem_bounds = manual_problem_bounds, 
                manual_problem_ref_point = anti_ideal_point, 
            )
            
            # Update training points
            train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
            train_obj_true_qehvi = torch.cat([train_obj_true_qehvi, new_obj_true_qehvi])
            
            # Compute hypervolume
            bd = DominatedPartitioning(ref_point = anti_ideal_point, Y = train_obj_true_qehvi)
            volume = bd.compute_hypervolume().item()
            hvs_qehvi.append(volume)

            # Reinitialize the models so they are ready for fitting on next iteration
            mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_true_qehvi)
            
            t1 = time.time()
            
            if verbose:
                write_to_log_file(log_file, f"\nBatch {iteration:>2}: Hypervolume (qEHVI) = ")
                write_to_log_file(log_file, f"({hvs_qehvi[-1]:>4.2f}), ")
                write_to_log_file(log_file, f"time = {t1-t0:>4.2f}. \n")
            else:
                write_to_log_file(log_file, ".")

            iteration += 1
    except Exception as e:
        log_exception(log_file, e)
    finally:
        log_file.close()

if __name__ == "__main__":
    main()
    