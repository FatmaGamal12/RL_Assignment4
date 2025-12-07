from environment import make_env

env = make_env("CarRacing-v3", render_mode="human")
obs, info = env.reset()

for _ in range(600):
    action = env.sample_action()
    obs, reward, done, trunc, info = env.step(action)
    if done or trunc: break
env.close()
