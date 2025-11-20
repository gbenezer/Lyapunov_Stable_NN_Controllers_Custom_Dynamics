#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1 # number of nodes ()
#SBATCH --gres=gpu:v100-sxm2:1 # type of GPU
#SBATCH --time=02:00:00
#SBATCH --job-name=pendulum_output_v100-sxm2_symbolic
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=output/benezerg/pendulum_output/job_logs/pendulum_output_v100-sxm2_symbolic.%j.out
#SBATCH --error=output/benezerg/pendulum_output/job_logs/pendulum_output_v100-sxm2_symbolic.%j.err

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"
python examples/pendulum_output_training_symbolic.py user=benezerg_pendulum_output_training_sxm2