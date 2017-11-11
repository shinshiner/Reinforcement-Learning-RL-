import gym
import gym_tetris
import time
import numpy as np

class Tetris():
	WIDTH = 50
	HEIGHT = 100
	def __init__(self, rank):
		self.env = gym.make('Tetris-v0')
		self.obs = None
		self.rank = rank
		self.life_time = 0

	def reset(self):
		self.obs = self.env.reset()
		#self.obs = self.reverse()
		self.obs = np.array([self.obs, ])
		self.life_time = 0
		return self.obs

	def step(self, action):
		self.obs, reward, done, info = self.env.step(action + 1)
		#self.obs = self.reverse()
		self.obs = np.array([self.obs, ])
		self.life_time += 1
		#time.sleep(0.1)
		return self.obs, done, reward, info

	def reverse(self):
		self.obs = self.obs.tolist()
		tmp = [[[0 for col in range(self.HEIGHT)] for row in range(self.WIDTH)] for channel in range(3)]
		for k in range(3):
			for i in range(self.WIDTH):
				for j in range(self.HEIGHT):
					tmp[k][i][j] = self.obs[i][j][k]
		#tmp = np.array([tmp])
		return tmp