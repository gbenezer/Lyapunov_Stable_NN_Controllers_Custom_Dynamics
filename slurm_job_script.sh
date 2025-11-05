#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1 # number of nodes ()
#SBATCH --gres=gpu:v100-pcie:1 # type of GPU
#SBATCH --time=04:00:00
#SBATCH --job-name=test_quadrotor_job_output
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=test_quadrotor_job_output.%j.out
#SBATCH --error=test_quadrotor_job_output.%j.err
#SBATCH --mail-user=benezer.gi@northeastern.edu
#SBATCH --mail-type=ALL

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1
conda activate lyapunov_neural_control
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"
python examples/quadrotor2d_output_training.py user=benezerg_quadrotor2d_state_training