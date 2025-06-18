# Deep Reinforcement Learning for Vision-Based Game Agents

This repository provides a modular and extensible framework for training and evaluating vision-based reinforcement learning (RL) agents in simulated game environments. The project supports multiple algorithms and is designed for image-based learning scenarios such as first-person games, mazes, or combat simulations.

## Overview

The goal is to train intelligent agents that can interact with visual game environments and learn optimal policies using deep reinforcement learning techniques. The agent receives pixel-based inputs and selects actions to maximize cumulative rewards over time.

Supported learning algorithms:
- Deep Q-Network (DQN)
- Advantage Actor-Critic (A2C)
- Proximal Policy Optimization (PPO)

## Features

- Vision-based RL using grayscale or RGB input
- Support for Gym-compatible environments (e.g., ViZDoom, Atari)
- Modular training scripts per algorithm
- Pretrained model loading and evaluation
- Configurable training parameters (learning rate, steps, reward shaping)
- Logging with TensorBoard for training monitoring

## Directory Structure

```
.
├── src/
│   ├── environment.py           # Game environment setup
│   ├── train.py                 # PPO training
│   ├── train_a2c.py             # A2C training
│   ├── train_dqn_sb3.py         # DQN training using Stable Baselines3
│   ├── test.py                  # Evaluation script
│   ├── test_a2c.py              # A2C agent testing
│   ├── test_dqn.py              # DQN agent testing
│   ├── models/                  # Custom model architectures
│   └── _vizdoom.ini             # Example VizDoom config
├── configs/                     # Training parameters and environment configs
├── scenarios/                   # Scenario scripts for visual environments
├── logs/                        # TensorBoard logs
├── models/                      # Saved agent checkpoints
└── README.md
```

## Installation

Install required packages using pip:

```bash
pip install torch torchvision stable-baselines3 gym opencv-python numpy matplotlib tensorboard
```

Key dependencies:
- `gym`
- `opencv-python`
- `stable-baselines3`
- `numpy`, `matplotlib`, `torch`
- `tensorboard`

## Supported Environments

Compatible with any Gym-style visual environment. Examples include:
- ViZDoom
- Atari
- Custom PyBullet or Unity setups

Ensure the environment provides:
- Frame-based observations
- Discrete or continuous action space
- A reward function

## Usage

### Train a PPO agent:
```bash
python src/train.py
```

### Train an A2C agent:
```bash
python src/train_a2c.py
```

### Train a DQN agent (Stable Baselines3):
```bash
python src/train_dqn_sb3.py
```

### Evaluate a trained agent:
```bash
python src/test.py --model models/your_model.zip
```

## Output

- Saved models in `models/`
- TensorBoard logs in `logs/`
- Plots of rewards and episode metrics

To visualize training:
```bash
tensorboard --logdir logs/
```

## Customization

You can modify:
- Network architectures (`src/models/`)
- Scenario difficulty and layout (`scenarios/`)
- Training hyperparameters (`configs/` or inside training scripts)
- Frame resolution, frame stacking, grayscale settings

## Author

**Faisal Hussain**  
LinkedIn: [linkedin.com/in/raifaisalhussain](https://www.linkedin.com/in/raifaisalhussain)
