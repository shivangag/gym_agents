import sys
import time
import random
import gym
import numpy as np
import tensorflow as tf
from collections import deque

class DQN(object):

	"""docstring for DQN"""
	def __init__(self, action_size, learning_rate):
		self.action_size = action_size
		self.hidden2_size = 10
		self.learning_rate = learning_rate
		self.obs = tf.placeholder(tf.float32, [None, 84, 80, 4])
		self.conv1 = tf.layers.conv2d(inputs=self.obs,
										filters=16,
										kernel_size=[8, 8],
										strides=(4, 4),
										activation=tf.nn.relu)
		self.conv2 = tf.layers.conv2d(inputs=self.conv1,
										filters=32,
										kernel_size=[4, 4],
										strides=(2, 2),
										activation=tf.nn.relu)
		self.conv2_flat = tf.reshape(self.conv2, [-1, 9 * 8 * 32])
		self.dense1 = tf.layers.dense(inputs=self.conv2_flat, units=256, activation=tf.nn.relu)
		self.q_val = tf.layers.dense(inputs=self.dense1, units=self.action_size)
		self.saver = tf.train.Saver()
		self.q_placeholder = tf.placeholder(tf.float32, [None, self.action_size])
		self.mse_loss = tf.losses.mean_squared_error(self.q_placeholder, self.q_val)
		self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mse_loss)


	def predict(self, obs, sess):
		return sess.run(self.q_val, {self.obs: obs})

	def fit(self, obs, q_val, sess):
		sess.run(self.train, {self.obs: obs, self.q_placeholder: q_val})

	def save(self, sess, path):
		save_path = self.saver.save(sess, path)
		print("Model saved in file: %s" % save_path)

	def restore(self, sess, path):
		self.saver.restore(sess, path)


class DQNAgent(object):
	"""docstring for DQNAgent"""
	def __init__(self, action_size):
		self.action_size = action_size
		self.memory = deque(maxlen=1000000)
		self.pcs = np.zeros((10, 84, 80, 4))
		self.batch_size = 32
		self.epsilon = 1.
		self.epsilon_min = 0.1
		self.epsilon_decay = 1e-6
		self.gamma = 0.95
		self.learning_rate = 0.0008
		self.model = DQN(self.action_size, self.learning_rate)
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
		states = np.zeros((32, 84, 80, 4))
		q_vals = np.zeros((32, 6))
		i = 0
		for state, action, reward, next_state, done in minibatch:
			target = reward
			states[i] = state
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state, self.sess))
			q_vals[i] = self.model.predict(state, self.sess)
			q_vals[i][action] = target
			i = i + 1
		self.model.fit(states, q_vals, self.sess)

	def preprocess(self, state):
		grayscale = state[0][: ,:, 1]		
		downscale = grayscale[None, ::2, ::2]
		processed_state =  downscale[:, 14:98, :, None]
		for i in range(3):
			grayscale = state[i + 1][: ,:, 1]
			downscale = grayscale[None, ::2, ::2]
			processed_state =  np.concatenate((downscale[:, 14:98, :, None], processed_state[:, :, :, :]), axis=3)
		return processed_state

	def save(self, path):
		self.model.save(self.sess, path)

	def restore(self, path):
		self.model.restore(self.sess, path)
	
	def estimate_progress(self):
		start_time = time.time()
		sum_q_val = 0
		sum_q_val = np.sum(np.max(agent.model.predict(self.pcs, self.sess)))
		print("Est time: {}".format(time.time() - start_time))
		return sum_q_val/10


if __name__ == "__main__":
	if(len(sys.argv) != 4):
		print("Usage: python3 dqn.py game_name train/play checkpoint_path")
	env = gym.make(sys.argv[1])
	action_size = env.action_space.n
	agent = DQNAgent(action_size)
	done = False
	if(sys.argv[2] == "train"):

		observation = env.reset()
		state = deque(maxlen=4)
		state.append(observation)
		j = 0
		for i in range(1,99):
			observation, reward, done, _ = env.step(env.action_space.sample())
			state.append(observation)
			if(i%9==0):
				processed_state = agent.preprocess(state)
				agent.pcs[j, :, :, :] = processed_state
				j = j + 1
	
		for e in range(30000):
			start_time = time.time()
			observation = env.reset()
			state.append(observation)
			for _ in range(15):
				observation, reward, done, _ = env.step(env.action_space.sample())
				state.append(observation)
			processed_state = agent.preprocess(state)
			for t in range(1000):
				action = agent.act(processed_state)
				next_observation, reward, done, _ = env.step(action)				
				state.append(next_observation)
				next_processed_state = agent.preprocess(state)
				agent.remember(processed_state, action, reward, next_processed_state, done)
				processed_state = next_processed_state
				if len(agent.memory) > agent.batch_size:
					agent.replay()
			if agent.epsilon > agent.epsilon_min:
				agent.epsilon -= agent.epsilon_decay
			if(e % 100 == 0):
				print("episode: {}/{}, avg_q_val: {}, e: {:.10}"
				  .format(e, 30000, agent.estimate_progress(), agent.epsilon))
				agent.save(sys.argv[3])
			print("Total time: {}".format(time.time() - start_time))
	else:
		observation = env.reset()
		state = deque(maxlen=4)
		state.append(observation)
		for _ in range(15):
			observation, reward, done, _ = env.step(env.action_space.sample())
			state.append(observation)
		processed_state = agent.preprocess(state)
		agent.restore(sys.argv[3])
		agent.epsilon = agent.epsilon_min
		num_state = 0
		while not done:
			env.render()
			action = agent.act(processed_state)
			next_observation, reward, done, _ = env.step(action)
			num_state += 1
			print(num_state)
			state.append(next_observation)
			next_processed_state = agent.preprocess(state)
			processed_state = next_processed_state
