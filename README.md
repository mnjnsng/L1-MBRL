# Robust Model Based Reinforcement Learning


We modified [this repo](https://github.com/WilsonWangTHU/mbbl-metrpo), which is the repo shared by the authors of MBBL-METRPO


## Results
<p align=center>
<img src="Results/data/METRPO_final_results.png" width=500>
</p>
<p align=center>
<img src="Results/data/Result_Table.PNG" width=500>
</p>


## Prerequisites
You need a MuJoCo license, and download MuJoCo 1.31. from 
https://www.roboti.us/. 
Useful information for installing MuJoCo can be found at 
https://github.com/openai/mujoco-py.

## Create a Conda environment
It's recommended to create a new Conda environment for this repo:

```
conda create -n <env_name> python=3.5
```
Or you can use python 3.6.

## Install package dependencies

```
pip install -r requirements.txt
```

Then please go to [MBBL](https://github.com/WilsonWangTHU/mbbl) to install the mbbl package for the environments.

## Changing Hyper-parameters

### MBRL Parameters 
This repo is based on the MBBl-METRPO repo. And therefore we use the same hyper-parameters / arguments system. We recommend looking at their repo or sample scripts under ./demo_scripts.

### L1 Adaptive control Parameters
The cutoff frequency and the switching threshold can be modified in `./libs/misc/data_handling/rollout_sampler.py`

## Run other experiments
Run experiments using the following command:

```python main.py --env <env_name> --exp_name <experiment_name> --sub_exp_name <exp_save_dir>```

- `env_name`: one of `(half_cheetah, ant, hopper, swimmer)`
- `exp_name`: what you want to call your experiment
- `sub_exp_name`: partial path for saving experiment logs and results
- `l1_ac`: 1 for switching on the adaptive controller else 0
- `test`: [0, 0] first argument for testing and second for the adaptive controller

Experiment results will be logged to `./experiments/<exp_save_dir>/<experiment_name>`

e.g. `python main.py --env invertedPendulum --exp_name test-exp --sub_exp_name test-exp-dir --l1_ac 1`


## Change configurations
You can modify the configuration parameters in `configs/params_<env_name>.json`.
