from collections import deque


class Environment:
	def __init__(self, env):
		self.env = env
		self.x_history = deque([-1], maxlen=60)
		self.high_score = 0
		self.done = False
		self.total_reward = 0


	def reset(self):
		self.x_history = deque([-1], maxlen=60)
		self.high_score = 0
		self.done = False
		self.total_reward = 0

		return self.env.reset()


	def step(self, action):
		state, reward, terminated, truncated, info = self.env.step(action)
		self._update_high_score(info['x_pos'])
		self.x_history.append(info['x_pos'])
		
		return state, reward, terminated, truncated, info


	def _not_moved(self):
		return len(set(self.x_history)) <= 1


	def _update_high_score(self, x_pos):
		if x_pos > self.high_score:
			self.high_score = x_pos


	def step_n_times(self, action, n):
	    reward_history = []

	    for _ in range(n):
	        next_frame, reward, terminated, truncated, _ = self.step(action)
	        reward_history.append(reward)

	        if terminated or truncated or self._not_moved():
	            self.done = True
	            break

	    avg_reward = sum(reward_history) / len(reward_history)

	    self.total_reward += avg_reward

	    return next_frame, avg_reward, self.done