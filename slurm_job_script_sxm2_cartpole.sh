#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1 # number of nodes ()
#SBATCH --gres=gpu:v100-sxm2:1 # type of GPU
#SBATCH --time=08:00:00
#SBATCH --job-name=cartpole_state_v100-sxm2_symbolic
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=output/benezerg/cartpole_state/job_logs/cartpole_state_v100-sxm2_symbolic.%j.out
#SBATCH --error=output/benezerg/cartpole_state/job_logs/cartpole_state_v100-sxm2_symbolic.%j.err

# prior log files need to be moved from pendulum_output
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"
python examples/cartpole_state_training_symbolic.py
