# Testing Guide for RL Assignment 4

## Overview

The `test.py` script allows you to evaluate trained RL agents on Box2D environments (LunarLander-v3, CarRacing-v3) and collect performance statistics.

## Prerequisites

Before testing, you must have a trained model. Train a model using:

```bash
python src/train.py --algorithm ppo --environment LunarLander-v3 --seed 42
```

This will create a model file at: `models/ppo_LunarLander-v3_seed42.pth`

## Basic Usage

### Test a trained model (default 100 episodes):

```bash
python src/test.py --algorithm ppo --environment LunarLander-v3
```

### Test with custom number of episodes:

```bash
python src/test.py --algorithm sac --environment LunarLander-v3 --num-episodes 50
```

### Test with rendering (visualize the agent):

```bash
python src/test.py --algorithm td3 --environment LunarLander-v3 --render --num-episodes 10
```

### Test with video recording:

```bash
python src/test.py --algorithm ppo --environment CarRacing-v3 --record-video --video-dir test_videos/
```

### Test and save results (creates plots and CSV files):

```bash
python src/test.py --algorithm sac --environment LunarLander-v3 --save-results
```

## Command-Line Arguments

| Argument         | Description                                | Default                                   |
| ---------------- | ------------------------------------------ | ----------------------------------------- |
| `--algorithm`    | Algorithm to test (td3, sac, ppo)          | **Required**                              |
| `--environment`  | Environment (LunarLander-v3, CarRacing-v3) | **Required**                              |
| `--model-path`   | Path to trained model                      | `models/{algorithm}_{env}_seed{seed}.pth` |
| `--num-episodes` | Number of test episodes                    | 100                                       |
| `--seed`         | Random seed for reproducibility            | 42                                        |
| `--render`       | Render environment during testing          | False                                     |
| `--record-video` | Record videos of test episodes             | False                                     |
| `--video-dir`    | Directory for recorded videos              | `test_videos/`                            |
| `--save-results` | Save results, plots, and CSV files         | False                                     |

## Output

### Console Output

The script displays:

- Configuration information
- Progress updates during testing
- Final statistics (mean, std, min, max rewards)

Example output:

```
============================================================
TEST RESULTS
============================================================
mean_reward: 245.3421
std_reward: 23.4567
min_reward: 180.2345
max_reward: 298.7654
============================================================
```

### Saved Results (when using `--save-results`)

The script creates a `results/` directory with:

1. **Text file**: `{algorithm}_{environment}_test_results.txt`

   - Summary statistics
   - Model path and configuration

2. **CSV file**: `{algorithm}_{environment}_test_episodes.csv`

   - Episode-by-episode data
   - Columns: Episode, Reward, Duration (if available)

3. **Plots**:
   - `{algorithm}_{environment}_test_rewards.png` - Box plot and histogram of rewards
   - `{algorithm}_{environment}_test_durations.png` - Episode durations (if available)

## Examples

### Example 1: Quick test with visualization

```bash
python src/test.py --algorithm sac --environment LunarLander-v3 --render --num-episodes 5
```

### Example 2: Comprehensive evaluation with all outputs

```bash
python src/test.py --algorithm ppo --environment LunarLander-v3 --num-episodes 100 --save-results --record-video
```

### Example 3: Test a specific model checkpoint

```bash
python src/test.py --algorithm td3 --environment CarRacing-v3 --model-path custom_models/best_model.pth --num-episodes 50
```

### Example 4: Test with different seed

```bash
python src/test.py --algorithm sac --environment LunarLander-v3 --seed 123 --model-path models/sac_LunarLander-v3_seed123.pth
```

## Troubleshooting

### Error: Model file not found

Make sure you've trained the model first:

```bash
python src/train.py --algorithm ppo --environment LunarLander-v3 --seed 42
```

### Error: test() or evaluate() method not implemented

Implement the testing method in your algorithm file (ppo.py or td3.py):

```python
def test(self, env, num_episodes=10):
    # TODO: Implement testing loop
    pass
```

### Error: load() method not implemented

Implement the load method in your algorithm file (already provided in the skeleton).

## Testing Workflow

1. **Train a model**:

   ```bash
   python src/train.py --algorithm ppo --environment LunarLander-v3
   ```

2. **Quick test** (5 episodes with rendering):

   ```bash
   python src/test.py --algorithm ppo --environment LunarLander-v3 --render --num-episodes 5
   ```

3. **Full evaluation** (100 episodes with statistics):

   ```bash
   python src/test.py --algorithm ppo --environment LunarLander-v3 --num-episodes 100 --save-results
   ```

4. **Review results** in the `results/` directory

## Notes

- The test script uses deterministic actions (no exploration noise) for evaluation
- All random seeds are set for reproducibility
- Video recording requires `rgb_array` render mode (automatically handled)
- SAC algorithm already has a complete test implementation you can use as reference
- PPO and TD3 test methods need to be implemented following the TODOs
