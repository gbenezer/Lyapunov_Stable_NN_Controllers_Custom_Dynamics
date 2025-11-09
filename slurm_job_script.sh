#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1 # number of nodes ()
#SBATCH --gres=gpu:v100-pcie:1 # type of GPU
#SBATCH --time=04:00:00
#SBATCH --job-name=quadrotor2d_output_08Nov2025_V100_PCIe_set_1
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=output/benezerg/quadrotor2d_output/job_logs/quadrotor2d_output_08Nov2025_V100_PCIe_set_1.%j.out
#SBATCH --error=output/benezerg/quadrotor2d_output/job_logs/quadrotor2d_output_08Nov2025_V100_PCIe_set_1.%j.err

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"
for i in {1..6};
do
    python examples/quadrotor2d_output_training.py user=benezerg_quadrotor2d_output_training
done