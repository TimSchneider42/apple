# Apple: Toward General Active Perception via Reinforcement Learning

This is the official code release for our [ICLR 2026 paper "Apple: Toward General Active Perception via Reinforcement Learning"](https://timschneider42.github.io/apple/).
To reproduce the results in our paper, please follow the instructions below.

## Installation

Follow these steps to set up the environment and install the necessary dependencies:

1. Install `uv`, following the instructions in the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

2. Clone the repository:

```bash
git clone https://github.com/TimSchneider42/apple/
```

3. Navigate to the project directory:

```bash
cd apple
```

4. Install the dependencies using `uv`:

```bash
uv sync --frozen
```

This code was developed and tested using Python 3.11 on Ubuntu 22.04.
Technically, it should work on other operating systems and Python versions, but we have not tested it on those.

## Running Experiments

To run experiments, best use the `run_experiment.bash` helper script:

```bash
# Optional: set this to your Weights & Biases entity if you want to log your experiments there
export WANDB_ENTITY=...

# Arguments:
# CONFIG:   The configuration file for the algorithm, e.g. "vit/sac=sac", see a full list below.
# ENV:      The environment to train on, e.g. "TactileMNIST-v0", see a full list below.
# EVAL_ENV: Optional: The corresponding evaluation environment, e.g. "TactileMNIST-test-v0, if
#           applicable. If not provided, the training environment will be used for evaluation as well.
# ARGS:     Optional: Additional arguments to pass to the training script, e.g. to specify hyperparameters.
./run_experiment.bash CONFIG ENV EVAL_ENV ARGS

# Example: Run APPLE-SAC on TactileMNIST-v0, evaluating on TactileMNIST-test-v0
./run_experiment.bash vit/sac=sac tactile_mnist:TactileMNIST-v0 tactile_mnist:TactileMNIST-test-v0

# Example: Run APPLE-CrossQ on CircleSquare-v0 for 1M steps
./run_experiment.bash dense/sac=crossq CircleSquare-v0 null algorithm.total_env_steps=1000000
```

### Available algorithm configurations

In the following is a list of all available configurations for the algorithms, which can be used as the `CONFIG` argument when running experiments.

#### Configurations for vision-based tactile tasks

All algorithm configurations in this table use a ViT as vision encoder, and are applicable to vision-based tactile tasks such as TactileMNIST, Toolbox, and TactileMNISTVolume.

| Configuration        | Description                                  |
|----------------------|----------------------------------------------|
| `vit/sac=sac`        | APPLE-SAC, the SAC-based APPLE variant.      |
| `vit/sac=crossq`     | APPLE-CrossQ, the CrossQ-based APPLE variant |
| `vit/sac=random_act` | APPLE-RND, taking random actions.            |

#### Configurations for non-vision-based tasks

All algorithm configurations in this table do not use a vision encoder, and are applicable to non-vision-based tasks such as CircleSquare, CIFAR10, and the HAMTactileClassification task (called MHSB in the paper).

| Configuration              | Description                                                                                                                              |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `dense/sac=sac`            | APPLE-SAC, the [SAC](https://arxiv.org/abs/1801.01290)-based APPLE variant.                                                              |
| `dense/sac=crossq`         | APPLE-CrossQ, the [CrossQ](https://arxiv.org/abs/1902.05605)-based APPLE variant                                                         |
| `dense/ppo=ppo`            | APPLE-PPO, the [PPO](https://arxiv.org/abs/1707.06347)-based APPLE variant                                                               |
| `dense/ham=ham`            | [HAM](https://arxiv.org/abs/1902.07501), a re-implementation of the Haptic Attention Module.                                             |
| `dense/ham=ham_cl`         | HAM modified to support arbitrary loss functions instead of just cross-entropy classification loss.                                      |
| `dense/sac=random_act`     | APPLE-RND, taking random actions.                                                                                                        |
| `dense/sac=grid_policy`    | APPLE-GRID, following a fixed grid search pattern.                                                                                       |
| `dense/sac=sac_lstm`       | APPLE-SAC-LSTM, using an LSTM instead of a transformer.                                                                                  |
| `dense/sac=crossq_lstm`    | APPLE-CrossQ-LSTM, using an LSTM instead of a transformer.                                                                               |
| `dense/sac=ppo_lstm`       | APPLE-PPO-LSTM, using an LSTM instead of a transformer.                                                                                  |
| `dense/sac=sac_pure_rl`    | APPLE-SAC-PURE-RL, treating the environment as a regular RL environment, assuming no knowledge of the loss function and target label.    |
| `dense/sac=crossq_pure_rl` | APPLE-CrossQ-PURE-RL, treating the environment as a regular RL environment, assuming no knowledge of the loss function and target label. |

All experimental data produced for the paper is available in the `data` directory.

### Tested environments

We tested our algorithms on the following environments, which can be used as the `ENV`/`EVAL_ENV` argument when running experiments.

| Environment                           | Description                                                                                                                                                                                                                 | Evaluation Environment                     | Vision-based Tactile? |
|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|-----------------------|
| `tactile_mnist:TactileMNIST-v0`       | A vision-based tactile environment from the [TMBS](https://timschneider42.github.io/tactile-mnist/) in which the agent has to classify 3D MNIST digits.                                                                     | `tactile_mnist:TactileMNIST-test-v0`       | Yes                   |
| `tactile_mnist:Toolbox-v0`            | A vision-based tactile environment from the TMBS in which the agent has to find to pose of a wrench on a plate.                                                                                                             | N/A                                        | Yes                   |
| `tactile_mnist:TactileMNISTVolume-v0` | A vision-based tactile environment from the TMBS in which the agent has determine the volume of 3D MNIST digits.                                                                                                            | `tactile_mnist:TactileMNISTVolume-test-v0` | Yes                   |
| `CircleSquare-v0`                     | A non-vision-based environment from [ap_gym](https://github.com/TimSchneider42/active-perception-gym) in which the agent has to move a glimpse around an image and determine whether the image contains a circle or square. | N/A                                        | No                    |
| `CIFAR10-v0`                          | A non-vision-based environment from ap_gym in which the agent has to move a glimpse around an image and determine the image's class among the 10 CIFAR10 classes.                                                           | N/A                                        | No                    |

More details of these environments can be found in the paper and in the documentation of the [Tactile MNIST Benchmark Suite](https://timschneider42.github.io/tactile-mnist/) and [ap_gym](https://github.com/TimSchneider42/active-perception-gym).

### Custom configurations

By passing the `ARGS` argument when running experiments, you can also specify custom hyperparameters and configurations.
In this project, we use [Hydra](https://hydra.cc/) for configuration management, which allows you to easily override any configuration value from the command line.
For a documentation of all hyperparameters and configuration options, please refer to the `config` directory.
Each subdirectory in `config` corresponds to one module of the algorithm and contains either `default.yaml` or `base.yaml` files which specify the default values for the hyperparameters of that module and document them.
Also, running
```bash
./run_experiment.bash CONFIG ENV EVAL_ENV ARGS --cfg=job

# For example
./run_experiment.bash vit/sac=sac tactile_mnist:TactileMNIST-v0 tactile_mnist:TactileMNIST-test-v0 algorithm.total_env_steps=1000000 --cfg=job
```
will cause Hydra to print the full configuration used for that experiment, which can be helpful to understand which hyperparameters are being used and to debug custom configurations.

### Note about differences in performance

In the paper, there was a bug in the code which made APPLE worse on some tasks than it was supposed to be.
In this implementation, we fixed the bug, meaning that APPLE might perform better on some tasks than it did in the paper.

## Citing

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{schneider2026apple,
    title = {APPLE: Toward General Active Perception via Reinforcement Learning},
    author = {Schneider, Tim and de Farias, Cristiana and Calandra, Roberto and Chen, Liming and Peters, Jan},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year = {2026}
}
```
