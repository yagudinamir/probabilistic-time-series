# Reliable Decisions with Threshold Calibration

This is a repository for reproducing the experiments from Reliable Decisions with Threshold Calibration.

## Reproducibility

We provide the commands to reproduce the UCI regression experiments from the paper. Commands for training the baseline UCI models can be found in `baseline_model_training` and commands for the recalibration experiments can be found in `recalibration_exp`. To run all the UCI experiments, follow the commands to setup the environments, download the appropriate datasets, create a results folder, and run the experiments.

```bash
cd point-calibration
conda env create -f environment.yml
conda activate calibration
python download_datasets.py
chmod +x run_experiments.sh
mkdir results
./run_experiments
```

We also provide the commands for reproducing the MIMIC-III and DHS Asset Wealth experiments. The MIMIC-III
dataset can be obtained upon request [here](https://physionet.org/content/mimiciii/1.4/). Commands for training the baseline (uncalibrated models) for these datasets can be found in `baseline_model_training_real_world`. Commands for the recalibration experiments can be found in `recalibration_exp_real_world`.
