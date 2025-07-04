from collections import deque
import random
import numpy as np


class Environment:
	def __init__(self, env):
		self.env = env
		self.x_history = deque([-1], maxlen=60)
		self.high_score = 0
		self.total_reward = 0
		self.done = False


	def reset(self):
		self.x_history = deque([-1], maxlen=60)
		self.high_score = 0
		self.total_reward = 0
		self.done = False

		return self.env.reset()


	def _step(self, action):
		state, reward, terminated, truncated, info = self.env.step(action)
		self._update_high_score(info['x_pos'])
		self.x_history.append(info['x_pos'])
		
		return state, reward, terminated, truncated, info


	def _not_moved(self):
		return len(set(self.x_history)) <= 1


	def _update_high_score(self, x_pos):
		if x_pos > self.high_score:
			self.high_score = x_pos


	def step(self, action, n=4):
		reward_history = []

		for _ in range(n):
			
			next_frame, reward, terminated, truncated, _ = self._step(action)
			reward_history.append(reward)

			if terminated or truncated or self._not_moved():
				self.done = True
				break

		avg_reward = sum(reward_history) / len(reward_history)
		avg_reward = np.clip(avg_reward, -1, 1)

		self.total_reward += avg_reward

		return next_frame, avg_reward, self.done


class Breakout:
	def __init__(self, env):
		self.env = env
		self.total_reward = 0
		self.high_score = 0
		self.lives = 0 


	def reset(self):
		obs, info = self.env.reset()

		self.total_reward = 0
		self.high_score = 0
		self.lives = info.get('lives', 0)
		
		return obs, info


	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)

		done = terminated or truncated
		
		# track current lives
		current_lives = info.get('lives', self.lives)
		if current_lives < self.lives:
			reward = -1
		self.lives = current_lives

		if done:
			reward = -1
		
		self.total_reward += reward
		if reward > 0:
			self.high_score += reward

		return obs, reward, done
