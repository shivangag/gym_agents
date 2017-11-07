import random
import gym
import numpy as np
import tensorflow as tf
from collections import deque

class DQN(object):

	"""docstring for DQN"""
	def __init__(self, observation_size, action_size, hidden_size, learning_rate):
		self.observation_size = observation_size
		self.action_size = action_size
		self.hidden_size = hidden_size
		self.learning_rate = learning_rate
		self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
		self.W1 = tf.get_variable("W1", [self.observation_size, self.hidden_size])
		self.b1 = tf.get_variable("b1", [self.hidden_size])
		self.hidden = tf.nn.relu(tf.matmul(self.obs, self.W1) + self.b1)
		self.W2 = tf.get_variable("W2", [self.hidden_size, self.action_size])
		self.b2 = tf.get_variable("b2", [self.action_size])
		self.q_val = tf.matmul(self.hidden, self.W2) + self.b2
		self.q_placeholder = tf.placeholder(tf.float32, [None, self.action_size])
		self.mse_loss = tf.losses.mean_squared_error(self.q_placeholder, self.q_val)
		self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mse_loss)


	def predict(self, obs, sess):
		return sess.run(self.q_val, {self.obs: obs})

	def fit(self, obs, q_val, sess):
		sess.run(self.train, {self.obs: obs, self.q_placeholder: q_val})



class DQNAgent(object):
	"""docstring for DQNAgent"""
	def __init__(self, observation_size, action_size):
		self.action_size = action_size
		self.observation_size = observation_size
		self.hidden_size = 24
		self.memory = deque(maxlen=2000)
		self.batch_size = 32
		self.epsilon = 0.9
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.997
		self.gamma = 0.95
		self.learning_rate = 0.002
		self.model = DQN(self.observation_size, self.action_size, self.hidden_size, self.learning_rate)
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		action_val = self.model.predict(state, self.sess)
		return np.argmax(action_val)

	def replay(self):
		minibatch = random.sample(self.memory, self.batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state, self.sess))
			q_vals = self.model.predict(state, self.sess)
			q_vals[0][action] = target
			self.model.fit(state, q_vals, self.sess)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
	env = gym.make('CartPole-v1')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = DQNAgent(state_size, action_size)
	done = False

	for e in range(1000):
		state = env.reset()
		state = np.reshape(state, [1, state_size])
		for time in range(500):
			env.render()
			action = agent.act(state)
			next_state, reward, done, _ = env.step(action)
			reward = reward if not done else -10
			next_state = np.reshape(next_state, [1, state_size])
			agent.remember(state, action, reward, next_state, done)
			state = next_state
			if done:
				print("episode: {}/{}, score: {}, e: {:.2}"
				      .format(e, 1000, time, agent.epsilon))
				break
		if len(agent.memory) > agent.batch_size:
			agent.replay()
