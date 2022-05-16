import numpy as np
import random
import math

# Generates grid mesh hyperparameter setting instances
# Code copied from https://github.com/amzn/differential-privacy-bayesian-optimization/tree/master/dpareto/grid_search
def generate_grid_hyperparameters_instances(hyperparam_ranges, points_per_param = 5, num_replications = 2):
    print("Generating hyperparameters mesh...", end='')
    linspaces = []
    param_names = []
    for param_name, range_descr in hyperparam_ranges.items():
        #  param_name = name of the parameter
        #  range_descr: dict = hyperparam_ranges[param_name]
        if 'value' in range_descr:
            # fixed value, attach to each instance after generation
            continue

        # Linearly spaced samples in the min - max range 
        # number_of_samples = points_per_param
        ls = np.linspace(range_descr['min'], range_descr['max'], points_per_param)
        linspaces.append(ls)
        param_names.append(param_name)

    # Creates rectangular grid over the search space of the hyperparamters
    # Combines all the samples of all the paramters
    # instances.shape = (points_per_param ** number_of_hyperparamters, number_of_hyperparamters)
    instances = np.array(np.meshgrid(*linspaces)).T.reshape(-1, len(linspaces))

    for param_name, range_descr in hyperparam_ranges.items():
        if 'value' in range_descr:
            column_to_add = np.full((instances.shape[0], 1), range_descr['value'])
            instances = np.hstack((instances, column_to_add))
            param_names.append(param_name)

    hyperparam_instances = []
    for instance in instances:
        hyperparam_instance = {}
        for i, param_name in enumerate(param_names):
            range_descr = hyperparam_ranges[param_name]
            if range_descr.get('round_to_int', False):
                hyperparam_instance[param_name] = int(round(instance[i]))
            else:
                hyperparam_instance[param_name] = instance[i]
        for j in range(num_replications):
            hyperparam_instances.append(hyperparam_instance)

    return hyperparam_instances

# Generates random hyperparameter setting instances
# Code copied from https://github.com/amzn/differential-privacy-bayesian-optimization/tree/master/dpareto/random_sampling
def generate_random_hyperparameter_instances(hyperparam_distributions, num_instances = 2, num_replications = 1):
    print("Generating random hyperparameters...", end='')
    hyperparam_instances = []
    for i in range(num_instances):
        # Generate a single random set of hyperparameters
        hyperparam_instance = {}
        for name, hyperparam_distribution in hyperparam_distributions.items():
            hyperparam_instance[name] = random_sample(hyperparam_distribution)

        for j in range(num_replications):
            hyperparam_instances.append(hyperparam_instance)

    print("Random hyperparameters generated.")
    return hyperparam_instances

def random_sample(distribution):
        distribution_type = distribution.get('type')

        reject_less_than = distribution.get('reject_less_than', False)
        reject_greater_than = distribution.get('reject_greater_than', False)
        if not reject_less_than:
            reject_less_than = -math.inf
        if not reject_greater_than:
            reject_greater_than = math.inf

        # Sample until number is within an acceptable range
        while True:
            if distribution_type == 'uniform':
                a = min(distribution['params'])
                b = max(distribution['params'])
                sample = random.uniform(a, b)
            elif distribution_type == 'normal':
                # sample from normal dist
                mu = distribution['params'][0]
                sigma = distribution['params'][1]
                sample = random.normalvariate(mu, sigma)
            elif distribution_type == 'exponential':
                # sample from exponential dist
                lambd = distribution['params'][0]
                shift = distribution['params'][1]
                sample = shift + random.expovariate(lambd)
            elif distribution_type == 'deterministic':
                sample = distribution['value']
            else:
                raise NotImplementedError("Must specify a supported sampling distribution.")

            if reject_less_than < sample < reject_greater_than:
                break

        # Optional: Round to nearest int
        if distribution.get('round_to_int', False):
            sample = int(round(sample))

        # Optional: Clip to bounds
        lower_bound = distribution.get('lower_bound', False)
        if lower_bound:
            sample = max(sample, lower_bound)
        upper_bound = distribution.get('upper_bound', False)
        if upper_bound:
            sample = min(sample, upper_bound)

        return sample