#!/bin/bash
#SBATCH --partition=atlas --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --exclude=atlas1,atlas2,atlas3,atlas4,atlas5,atlas6,atlas7,atlas8,atlas9,atlas10,atlas11,atlas12,atlas13,atlas14,atlas15,atlas20,atlas18,atlas17,atlas16
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="baseline_model_mimic_los_gaussian_laplace_mixture_nll_0.sh"
#SBATCH --error="/point-calibration/baseline_model_training_real_world/baseline_model_mimic_los_gaussian_laplace_mixture_nll_0_err.log"
#SBATCH --output="/point-calibration/baseline_model_training_real_world/baseline_model_mimic_los_gaussian_laplace_mixture_nll_0_out.log"

echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST

python /point-calibration/train_baseline_models.py main --seed 0 --loss gaussian_laplace_mixture_nll --save uncalibrated --epochs 100 --dataset mimic_los
sleep 1
