# CoCoIV: Towards Instrumental Variable Representation Learning under Confounded Covariates

## Description
This repository provides reproduction guidelines for the paper *Towards Instrumental Variable Representation Learning under Confounded Covariates*. It contains the source codes used in our empirical evaluations. We hope you find it valuable.

The main implementation of our model can be found in `Models/ours.py`.
The implementation of exploited variational encoders and decoders is based on:
- [Pytorch-VaDE by mperezcarrasco](https://github.com/mperezcarrasco/Pytorch-VaDE)
- [PyTorch-VAE by AntixK](https://github.com/AntixK/PyTorch-VAE/tree/master)

with modifications suitable for our architecture.

## Installation & Running
To get started, you may follow these instructions.

### Create envrironment
Create a new virtual environment by using `iv_rep.yaml`. Please use Python 3.9.18 since we did not test other versions. If you execute the command below, environment named 'iv' will be generated.
```bash
conda env create -f iv_rep.yaml
```
### Training the model
Once the environment is set up, you can either execute scripts to run the complete process 
    e.g. Run the main experiments for synthetic datasets:
  ```bash
  bash experiment_syn.sh
  ```
Here are the objectives for each script:
- Experiments for synthetic datasets: `experiment_syn.sh`
- Extended experiments (ablation, robustness, etc.): `experiment_extended.sh`
- Experiments for real-world datasets: `experiment_real.sh`

Results will be recorded in the `Result` folder.


### Testing with the trained models
We provide trained models for each experimental configuration in `./weights `. Each saved model is the best one based on training loss among 20 independent replications for its configuration. Thus, if you run the following commands, you can obtain 10 result files—one for each configuration (8 synthetic datasets + 2 real‐world datasets)—each containing a single row of estimator results obtained from the saved model (e.g. `TEST_ours_{treatment_type}_Syn_{dimension type}_{response function type}.csv`)
```bash
bash experiment_test.sh
```
Be sure that if you once train the model, the saved weights will be overwritten. Therefore, if you want to preserve the existing weights, we recommend saving them in a different directory.

## Details of Running Scripts
If you want to run experiments with personal settings, you may use the following command:
```bash
python main.py --exp_id syn --treatment b --response linear --highdim 1 --repetition 20 
```
- exp_id: type of experiments (syn: synthetic datasets)
- treatment: treatment type (binary (b) or continuous (con) 
- response: type of response function (linear / nonlinear)
- highdim: high-dimensional datasets or not ( high-dimensional (1) /low-dimension (0))
- repetition: number of experiments and range of random seed ( if 20, random seed may range from 0 to 19) )

For more arguments, you can refer to `main.py`.



## Architecture Choice
Variations of our architecture are available in:
- `Models/ours_vade.py` (VaDEs for encoders of Z, C)
- `Models/no_vade_at_all.py` (VAEs for encoders of Z, C)

