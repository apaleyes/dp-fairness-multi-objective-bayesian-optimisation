#!/bin/sh
cd ~
chmod +x /home/bf319/dp-fairness-multi-objective-bayesian-optimisation/pfairdp/run_hyperparametrised_pipeline.py
sbatch /home/bf319/dp-fairness-multi-objective-bayesian-optimisation/schedule-jobs/config2-dp-fairness_dir/slurm_cpu.peta4-icelake