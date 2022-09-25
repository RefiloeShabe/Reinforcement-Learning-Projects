#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
import numpy as np
from reinforce_tf2 import Agent
from utils import plotLearning


# In[ ]:


if __name__ == '__main__':
  agent = Agent(alpha = 0.0005, gamma = 0.99, n_actions = 4)
  env = gym.make('LunarLander-v2')

  score_history = []

  n_episodes = 2000

  for i in range(n_episodes):
    done = False
    score = 0
    observation = env.reset()

    while not done:
      action = agent.choose_action(observation)
      observation_, reward, done, info = env.step(action)
      observation = observation_
      score += reward
    score_history.append(score)

    agent.learn()

    avg_score = np.mean(score_histore[-100:])
    print('episode ', i, 'score %.1f' %score, 'avg score %.1f' % avg_score)

  filename = 'lunar_lander.png'
  plotLearning(score_history, filename = filename, window = 100)

