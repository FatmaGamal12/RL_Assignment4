from environment import make_env
import time

# we enable human rendering
env = make_env("LunarLander-v3", render_mode="human", record_video=False)

obs, info = env.reset()

for step in range(500):
    action = env.sample_action()  # random movement
    obs, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.02)  # slow down so we can see it visually

    if terminated or truncated:
        break

env.close()
