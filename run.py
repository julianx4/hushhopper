from hopperEnv import WalkingRobotEnv
from stable_baselines3 import PPO
import os

models_dir = "models/PPO"
logdir = "logs"

env = WalkingRobotEnv(GUI = True)
env.reset()

model_path = f"{models_dir}/1930000"
model_path = "models_to_keep/4380000"
model = PPO.load(model_path, env=env)


while True:
    obs,_ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, _, info = env.step(action)
        env.render()
        print(rewards)