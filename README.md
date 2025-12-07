# RL Assignment 4: Deep Reinforcement Learning on Box2D Environments

A comprehensive implementation of state-of-the-art deep reinforcement learning algorithms (PPO, SAC, TD3) for continuous control tasks in Box2D environments.

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Environments](#environments)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This project implements three advanced deep reinforcement learning algorithms for continuous action spaces:

- **PPO (Proximal Policy Optimization)** - On-policy actor-critic with clipped surrogate objective
- **SAC (Soft Actor-Critic)** - Off-policy maximum entropy RL with automatic temperature tuning
- **TD3 (Twin Delayed DDPG)** - Off-policy actor-critic with twin Q-networks and delayed updates

All algorithms are designed for **continuous action spaces** and tested on **Box2D environments** from Gymnasium.

---

## 🤖 Algorithms Implemented

### 1. PPO (Proximal Policy Optimization)

- **Type**: On-policy
- **Features**:
  - Clipped surrogate objective
  - Generalized Advantage Estimation (GAE)
  - Entropy regularization
  - Mini-batch updates
- **File**: `src/algorithms/ppo.py`
- **Status**: Skeleton with TODOs (needs implementation)

### 2. SAC (Soft Actor-Critic)

- **Type**: Off-policy
- **Features**:
  - Twin Q-networks (clipped double Q-learning)
  - Automatic entropy temperature tuning
  - Replay buffer
  - Target policy smoothing
- **File**: `src/algorithms/sac.py`
- **Status**: ✅ Fully implemented and ready to use

### 3. TD3 (Twin Delayed DDPG)

- **Type**: Off-policy
- **Features**:
  - Twin critic networks
  - Delayed policy updates
  - Target policy smoothing
  - Exploration noise
- **File**: `src/algorithms/td3.py`
- **Status**: Skeleton with TODOs (needs implementation)

---

## 🌍 Environments

### LunarLander-v3

- **Description**: Land a spacecraft between two flags
- **Observation Space**: Box(8) - position, velocity, angle, angular velocity, leg contact
- **Action Space**: Box(2) - main engine, lateral engine (continuous)
- **Difficulty**: Moderate
- **Target Score**: 200+

### CarRacing-v3

- **Description**: Race car around a track from a bird's eye view
- **Observation Space**: Box(96, 96, 3) - RGB image (normalized)
- **Action Space**: Box(3) - steering [-1,1], gas [0,1], brake [0,1]
- **Difficulty**: Hard
- **Target Score**: 900+

---

## 📁 Project Structure

```
RL_Assignment4/
├── configs/                      # Configuration files
│   ├── ppo_config.yaml          # PPO hyperparameters
│   ├── sac_config.yaml          # SAC hyperparameters
│   └── td3_config.yaml          # TD3 hyperparameters
├── src/
│   ├── algorithms/              # Algorithm implementations
│   │   ├── __init__.py
│   │   ├── ppo.py              # PPO algorithm
│   │   ├── sac.py              # SAC algorithm
│   │   └── td3.py              # TD3 algorithm
│   ├── utils/                   # Utility modules
│   │   ├── __init__.py
│   │   ├── logger.py           # W&B logging
│   │   └── plotting.py         # Visualization utilities
│   ├── environment.py           # Environment wrapper
│   ├── train.py                # Main training script
│   └── test.py                 # Testing/evaluation script
├── models/                      # Saved model checkpoints (created during training)
├── results/                     # Test results and plots (created during testing)
├── videos/                      # Recorded videos (optional)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── TESTING_GUIDE.md            # Detailed testing guide
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the repository

```bash
cd c:\Users\Aisha\Desktop\RL\RL_Assignment4
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Required packages:

- `torch` - Deep learning framework
- `gymnasium` - RL environments
- `numpy` - Numerical computations
- `matplotlib` - Plotting
- `pyyaml` - Configuration files
- `wandb` - Experiment tracking (optional)

---

## ⚡ Quick Start

### Train SAC on LunarLander (works out of the box):

```bash
python src/train.py --algorithm sac --environment LunarLander-v3
```

### Test the trained model:

```bash
python src/test.py --algorithm sac --environment LunarLander-v3 --render --num-episodes 10
```

---

## 🎓 Training

### Basic Training Command

```bash
python src/train.py --algorithm <ALGO> --environment <ENV>
```

### Training Options

| Argument          | Options                          | Description            | Default          |
| ----------------- | -------------------------------- | ---------------------- | ---------------- |
| `--algorithm`     | `ppo`, `sac`, `td3`              | Algorithm to use       | **Required**     |
| `--environment`   | `LunarLander-v3`, `CarRacing-v3` | Environment            | **Required**     |
| `--seed`          | Any integer                      | Random seed            | `42`             |
| `--device`        | `cpu`, `cuda`                    | Training device        | Auto-detect      |
| `--render`        | Flag                             | Render during training | `False`          |
| `--record-video`  | Flag                             | Record training videos | `False`          |
| `--video-dir`     | Path                             | Video directory        | `videos/`        |
| `--wandb-project` | String                           | W&B project name       | `rl-assignment4` |
| `--no-wandb`      | Flag                             | Disable W&B logging    | `False`          |

### Training Examples

#### Example 1: Train PPO on LunarLander

```bash
python src/train.py --algorithm ppo --environment LunarLander-v3 --seed 42
```

**Expected Output:**

```
============================================================
Training Configuration
============================================================
Algorithm: PPO
Environment: LunarLander-v3
Device: cuda
Seed: 42
...
============================================================
Starting Training
============================================================
Training for 1000 episodes...
```

**Output Files:**

- Model: `models/ppo_LunarLander-v3_seed42.pth`
- W&B logs (if enabled)

---

#### Example 2: Train SAC on CarRacing

```bash
python src/train.py --algorithm sac --environment CarRacing-v3 --seed 123
```

**Expected Output:**

```
============================================================
Training Configuration
============================================================
Algorithm: SAC
Environment: CarRacing-v3
Device: cuda
Seed: 123
...
```

**Output Files:**

- Model: `models/sac_CarRacing-v3_seed123.pth`

---

#### Example 3: Train TD3 with Video Recording

```bash
python src/train.py --algorithm td3 --environment LunarLander-v3 --record-video --video-dir training_videos/
```

**Output Files:**

- Model: `models/td3_LunarLander-v3_seed42.pth`
- Videos: `training_videos/rl-video-episode-*.mp4`

---

#### Example 4: Train without W&B logging

```bash
python src/train.py --algorithm sac --environment LunarLander-v3 --no-wandb
```

---

#### Example 5: Train with custom W&B project

```bash
python src/train.py --algorithm ppo --environment CarRacing-v3 --wandb-project my-rl-project
```

---

### All Training Combinations

#### PPO Training

```bash
# LunarLander
python src/train.py --algorithm ppo --environment LunarLander-v3

# CarRacing
python src/train.py --algorithm ppo --environment CarRacing-v3
```

#### SAC Training

```bash
# LunarLander
python src/train.py --algorithm sac --environment LunarLander-v3

# CarRacing
python src/train.py --algorithm sac --environment CarRacing-v3
```

#### TD3 Training

```bash
# LunarLander
python src/train.py --algorithm td3 --environment LunarLander-v3

# CarRacing
python src/train.py --algorithm td3 --environment CarRacing-v3
```

---

## 🧪 Testing

### Basic Testing Command

```bash
python src/test.py --algorithm <ALGO> --environment <ENV>
```

### Testing Options

| Argument         | Options                          | Description             | Default          |
| ---------------- | -------------------------------- | ----------------------- | ---------------- |
| `--algorithm`    | `ppo`, `sac`, `td3`              | Algorithm to test       | **Required**     |
| `--environment`  | `LunarLander-v3`, `CarRacing-v3` | Environment             | **Required**     |
| `--model-path`   | Path                             | Custom model path       | Auto (see below) |
| `--num-episodes` | Integer                          | Number of test episodes | `100`            |
| `--seed`         | Integer                          | Random seed             | `42`             |
| `--render`       | Flag                             | Render during testing   | `False`          |
| `--record-video` | Flag                             | Record test videos      | `False`          |
| `--video-dir`    | Path                             | Video directory         | `test_videos/`   |
| `--save-results` | Flag                             | Save results and plots  | `False`          |

**Default model path**: `models/{algorithm}_{environment}_seed{seed}.pth`

### Testing Examples

#### Example 1: Test PPO on LunarLander (basic)

```bash
python src/test.py --algorithm ppo --environment LunarLander-v3 --num-episodes 100
```

**Expected Output:**

```
============================================================
Testing Configuration
============================================================
Algorithm: PPO
Environment: LunarLander-v3
Model path: models/ppo_LunarLander-v3_seed42.pth
Number of test episodes: 100
...
============================================================
Running Test Episodes
============================================================
[TEST] Episode 1/100 | Reward: 245.32
[TEST] Episode 2/100 | Reward: 189.45
...
============================================================
TEST RESULTS
============================================================
mean_reward: 234.5678
std_reward: 45.1234
min_reward: 120.4567
max_reward: 298.7654
============================================================
```

---

#### Example 2: Test SAC with Visualization (watch 10 episodes)

```bash
python src/test.py --algorithm sac --environment LunarLander-v3 --render --num-episodes 10
```

This will show the agent performing in the environment in real-time.

---

#### Example 3: Test TD3 with Full Results Saved

```bash
python src/test.py --algorithm td3 --environment CarRacing-v3 --num-episodes 50 --save-results
```

**Output Files Created:**

- `results/td3_CarRacing-v3_test_results.txt` - Summary statistics
- `results/td3_CarRacing-v3_test_episodes.csv` - Episode-by-episode data
- `results/td3_CarRacing-v3_test_rewards.png` - Reward distribution plots

---

#### Example 4: Test with Video Recording

```bash
python src/test.py --algorithm sac --environment LunarLander-v3 --record-video --video-dir test_videos/ --num-episodes 5
```

**Output Files:**

- `test_videos/rl-video-episode-0.mp4`
- `test_videos/rl-video-episode-1.mp4`
- etc.

---

#### Example 5: Test Custom Model

```bash
python src/test.py --algorithm ppo --environment LunarLander-v3 --model-path custom_models/best_ppo.pth --num-episodes 20
```

---

#### Example 6: Test with Different Seed

```bash
# First train with seed 999
python src/train.py --algorithm sac --environment LunarLander-v3 --seed 999

# Then test it
python src/test.py --algorithm sac --environment LunarLander-v3 --seed 999 --num-episodes 100
```

---

### All Testing Combinations

#### PPO Testing

```bash
# LunarLander - Quick test (10 episodes with rendering)
python src/test.py --algorithm ppo --environment LunarLander-v3 --render --num-episodes 10

# LunarLander - Full evaluation (100 episodes with results)
python src/test.py --algorithm ppo --environment LunarLander-v3 --num-episodes 100 --save-results

# CarRacing - Quick test
python src/test.py --algorithm ppo --environment CarRacing-v3 --render --num-episodes 5

# CarRacing - Full evaluation
python src/test.py --algorithm ppo --environment CarRacing-v3 --num-episodes 100 --save-results
```

#### SAC Testing

```bash
# LunarLander - Quick test
python src/test.py --algorithm sac --environment LunarLander-v3 --render --num-episodes 10

# LunarLander - Full evaluation
python src/test.py --algorithm sac --environment LunarLander-v3 --num-episodes 100 --save-results

# CarRacing - Quick test
python src/test.py --algorithm sac --environment CarRacing-v3 --render --num-episodes 5

# CarRacing - Full evaluation
python src/test.py --algorithm sac --environment CarRacing-v3 --num-episodes 100 --save-results
```

#### TD3 Testing

```bash
# LunarLander - Quick test
python src/test.py --algorithm td3 --environment LunarLander-v3 --render --num-episodes 10

# LunarLander - Full evaluation
python src/test.py --algorithm td3 --environment LunarLander-v3 --num-episodes 100 --save-results

# CarRacing - Quick test
python src/test.py --algorithm td3 --environment CarRacing-v3 --render --num-episodes 5

# CarRacing - Full evaluation
python src/test.py --algorithm td3 --environment CarRacing-v3 --num-episodes 100 --save-results
```

---

## 📊 Results

### Results Directory Structure

When using `--save-results`, the following files are created in the `results/` directory:

```
results/
├── ppo_LunarLander-v3_test_results.txt      # Summary statistics
├── ppo_LunarLander-v3_test_episodes.csv     # Episode data
├── ppo_LunarLander-v3_test_rewards.png      # Reward plots
├── sac_CarRacing-v3_test_results.txt
├── sac_CarRacing-v3_test_episodes.csv
├── sac_CarRacing-v3_test_rewards.png
└── ...
```

### Understanding Results

#### Text File Format (`.txt`)

```
Test Results for SAC on LunarLander-v3
Model: models/sac_LunarLander-v3_seed42.pth
Number of episodes: 100
Seed: 42
============================================================

mean_reward: 245.3421
std_reward: 23.4567
min_reward: 180.2345
max_reward: 298.7654
```

#### CSV File Format (`.csv`)

```csv
Episode,Reward,Duration
1,245.32,234
2,189.45,187
3,278.91,312
...
```

#### Plot Files (`.png`)

- Box plot showing reward distribution
- Histogram showing frequency of rewards
- Statistics overlay (mean, std, min, max)

---

## ⚙️ Configuration

### Configuration Files Location

All hyperparameters are stored in YAML files:

- `configs/ppo_config.yaml`
- `configs/sac_config.yaml`
- `configs/td3_config.yaml`

### Example: PPO Configuration

```yaml
LunarLander-v3:
  episodes: 1000
  learning_rate: 0.0003
  discount_factor: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  entropy_coef: 0.01
  value_coef: 0.5
  max_grad_norm: 0.5
  batch_size: 64
  n_steps: 2048
  n_epochs: 10
  max_episode_steps: 1000

CarRacing-v3:
  episodes: 1500
  learning_rate: 0.00025
  # ... more parameters
```

### Modifying Hyperparameters

1. Open the appropriate config file (e.g., `configs/ppo_config.yaml`)
2. Edit the parameters under your environment
3. Save the file
4. Run training (changes are automatically loaded)

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Model Not Found Error

```
ERROR: Model file not found!
Looking for: models/ppo_LunarLander-v3_seed42.pth
```

**Solution:** Train the model first:

```bash
python src/train.py --algorithm ppo --environment LunarLander-v3 --seed 42
```

---

#### 2. Method Not Implemented Error

```
ERROR: Training method for PPO needs to be implemented.
```

**Solution:** The algorithm needs implementation. Check the TODO comments in:

- `src/algorithms/ppo.py` for PPO
- `src/algorithms/td3.py` for TD3

Use SAC (`src/algorithms/sac.py`) as a reference - it's fully implemented.

---

#### 3. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:** Use CPU instead:

```bash
python src/train.py --algorithm sac --environment LunarLander-v3 --device cpu
```

Or reduce batch size in the config file.

---

#### 4. W&B Login Required

```
wandb: ERROR Error while calling W&B API
```

**Solution:** Either login to W&B:

```bash
wandb login
```

Or disable W&B:

```bash
python src/train.py --algorithm sac --environment LunarLander-v3 --no-wandb
```

---

#### 5. Import Error

```
ModuleNotFoundError: No module named 'gymnasium'
```

**Solution:** Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📝 Implementation Status

| Algorithm | Training    | Testing     | Status                  |
| --------- | ----------- | ----------- | ----------------------- |
| **SAC**   | ✅ Complete | ✅ Complete | 🟢 Ready to use         |
| **PPO**   | 🔨 TODOs    | 🔨 TODOs    | 🟡 Needs implementation |
| **TD3**   | 🔨 TODOs    | 🔨 TODOs    | 🟡 Needs implementation |

### Next Steps for PPO and TD3

1. Open `src/algorithms/ppo.py` or `src/algorithms/td3.py`
2. Find methods marked with `TODO:`
   - `select_action()`
   - `train()`
   - `evaluate()` or `test()`
   - Other helper methods
3. Implement following the detailed TODO comments
4. Use SAC as a reference implementation

---

## 🎯 Recommended Workflow

### 1. Quick Start (using SAC)

```bash
# Train
python src/train.py --algorithm sac --environment LunarLander-v3

# Watch it play (5 episodes)
python src/test.py --algorithm sac --environment LunarLander-v3 --render --num-episodes 5

# Full evaluation
python src/test.py --algorithm sac --environment LunarLander-v3 --num-episodes 100 --save-results
```

### 2. Development Workflow (implementing PPO/TD3)

```bash
# 1. Implement the algorithm
# Edit src/algorithms/ppo.py or src/algorithms/td3.py

# 2. Test training (short run)
python src/train.py --algorithm ppo --environment LunarLander-v3 --no-wandb

# 3. If training works, run full training
python src/train.py --algorithm ppo --environment LunarLander-v3

# 4. Evaluate the trained model
python src/test.py --algorithm ppo --environment LunarLander-v3 --num-episodes 100 --save-results
```

### 3. Comparison Workflow

```bash
# Train all algorithms
python src/train.py --algorithm ppo --environment LunarLander-v3
python src/train.py --algorithm sac --environment LunarLander-v3
python src/train.py --algorithm td3 --environment LunarLander-v3

# Test all algorithms
python src/test.py --algorithm ppo --environment LunarLander-v3 --num-episodes 100 --save-results
python src/test.py --algorithm sac --environment LunarLander-v3 --num-episodes 100 --save-results
python src/test.py --algorithm td3 --environment LunarLander-v3 --num-episodes 100 --save-results

# Compare results in results/ directory
```

---

## 📚 References

- [PPO Paper](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [SAC Paper](https://arxiv.org/abs/1801.01290) - Haarnoja et al., 2018
- [TD3 Paper](https://arxiv.org/abs/1802.09477) - Fujimoto et al., 2018
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

---

## 👥 Authors

- **Assignment**: RL Assignment 4
- **Repository**: FatmaGamal12/RL_Assignment4

---

## 📄 License

This project is for educational purposes as part of an RL course assignment.

---

## 🆘 Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the `TESTING_GUIDE.md` for detailed testing instructions
3. Examine the SAC implementation as a working reference
4. Check the TODO comments in PPO and TD3 files for implementation guidance

---

**Happy Training! 🚀**
