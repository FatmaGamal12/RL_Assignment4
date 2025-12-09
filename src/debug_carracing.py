from environment import make_env

env = make_env("CarRacing-v3")

obs, info = env.reset()

print("Initial observation shape:", obs.shape)
print("First 10 values:", obs[:10])

for step in range(5):
    action = env.sample_action()
    next_obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nStep {step}")
    print("Observation shape:", next_obs.shape)
    print("Reward:", reward)

    if terminated or truncated:
        print("Episode ended early")
        break

env.close()
