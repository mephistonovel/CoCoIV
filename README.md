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

1. Create a new virtual environment by using either `iv_rep_package.txt` or `iv_rep.yaml`. Please use Python 3.9.18 since we did not test other versions. If you execute the command below, environment named 'iv' will be generated.
    ```bash
    conda env create -f iv_rep.yml
    ```

2. Once the environment is set up, you can either execute scripts to run the complete process 
    e.g. Run the main experiments for synthetic datasets:
      ```bash
      bash experiment_syn.sh
      ```
    Here are the objectives for each script:
    - Experiments for synthetic datasets: `experiment_syn.sh`
    - Extended experiments (ablation, robustness, etc.): `experiment_extended.sh`
    - Experiments for real-world datasets: `experiment_real.sh`

3. Results will be recorded in the `Result` folder.

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

You can refer to Appendix.C in the paper for more details. 

