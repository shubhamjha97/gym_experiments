import numpy as np
from time import time
import tensorflow as tf
import random
import os, sys
import matplotlib.pyplot as plt

class DQN:
	def __init__(self, env_, buffer_size = 10000, batch_size = 128):
		self.env_ = env_
		self.buffer_size = buffer_size
		self.buffer_ = []
		self.buffer_index = None
		self.action_space = self.env_.env.action_space
		self.num_actions = self.action_space.n
		self.state_size = self.env_.env.observation_space.shape[0]
		self.moving_reward = None
		self.gamma = 0.95
		self.eps = 0.01
		self.batch_size = batch_size
		self.moving_reward = None

		# DQN
		self.hidden1_size = 64
		self.hidden2_size = 32

		self.state_placeholder = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32, name='state')
		self.actions_placeholder = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32, name='actions')
		# self.rewards_placeholder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='rewards')
		# self.done_placeholder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='done_placeholder') # for detecting terminal states
		self.learning_rate = tf.placeholder(dtype=tf.float32, name='lr')
		self.targets = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='targets')

		self.hidden1 = tf.nn.relu(tf.layers.dense(self.state_placeholder, self.hidden1_size, kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None, name='hidden1'))
		self.hidden2 = tf.nn.relu(tf.layers.dense(self.hidden1, self.hidden2_size, kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None, name='hidden2'))
		self.q_values = tf.layers.dense(self.hidden2, self.num_actions, kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None, name='q_values')
		
		self.max_q_values = tf.reshape(tf.reduce_max(self.q_values, axis=1, name='max_q_values'), [-1,1])
		self.selected_q_values = tf.reshape(tf.reduce_sum(tf.multiply(self.q_values, self.actions_placeholder, name='selected_q_values'), axis=1), [-1,1])
		# self.targets = tf.add(self.rewards_placeholder, self.gamma*tf.multiply((1-self.done_placeholder), self.max_q_values), name='targets')

		with tf.name_scope("loss_fn"):
			self.loss = tf.reduce_mean(tf.square(tf.subtract(self.targets, self.selected_q_values)))
		
		self.optim_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		self.sess = tf.Session()
		self.summary_op = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter("tensorboard/dqn/", self.sess.graph)
		self.writer.close()
		self.sess.run(tf.global_variables_initializer())

	def train(self, env, episodes = 10, lr = 0.01, gamma = 0.95, eps = 0.01):
		self.gamma = gamma
		self.lr = lr
		self.eps = eps
		all_total_rewards=[]

		for episode in range(episodes):
			done = False
			obs = self.env_.reset()
			step = 0
			reward_sum = 0
			ep_start_time = time()
			while not done:
				step+=1
				temp = {}
				action = self.action(obs)
				temp['state'] = obs
				temp['action'] = action
				obs, reward, done, info = self.env_.step(action)
				reward_sum += reward
				temp['reward'] = reward
				temp['done'] = done
				# if episode>1000:
				# 	self.env_.env.render()
				self.store_experience(temp)
				if len(self.buffer_) > self.batch_size+1:
					self.update_net(self.lr)
			
			if self.moving_reward is None:
				self.moving_reward = reward_sum
			else:
				self.moving_reward = 0.99 * self.moving_reward + 0.01 * reward_sum

			all_total_rewards.append(reward_sum)
			print("Episode:", episode, "Steps:", step, "reward:", self.moving_reward, "lr", self.lr, "Time:", time()-ep_start_time)
			
		# Plot rewards
		fig = plt.figure()
		plt.plot(range(episodes), all_total_rewards)
		plt.xlabel("episode")
		plt.ylabel("total reward")
		fig.savefig("reward_curve_dqn.png")

	def store_experience(self, exp):
		if len(self.buffer_)>=self.buffer_size:
			if self.buffer_index is None:
				self.buffer_index = 0
			if self.buffer_index >= self.buffer_size:
				self.buffer_index = 0
			self.buffer_[self.buffer_index] = exp
			self.buffer_index+=1
		else:
			self.buffer_.append(exp)

	def action(self, state):
		if random.uniform(0,1) < self.eps:
			return random.sample(range(self.num_actions), 1)[0]
		q_values = self.sess.run(self.q_values, feed_dict={self.state_placeholder:np.array(state).reshape(1, -1)})
		action = np.argmax(q_values[0])
		return action

	def update_net(self, lr = 0.001):
		sampled_buffer = random.sample(self.buffer_, min(self.batch_size, len(self.buffer_)))
		states = np.array([x['state'] for x in sampled_buffer])
		rewards = np.array([x['reward'] for x in sampled_buffer]).reshape([-1, 1])
		done_arr = np.array([x['done'] for x in sampled_buffer]).reshape([-1, 1])

		actions = np.zeros([states.shape[0], self.num_actions])
		for i, x in enumerate(sampled_buffer):
			temp_action = x['action']
			actions[i, temp_action] = 1

		q_vals = self.sess.run(self.q_values, feed_dict={self.state_placeholder:states})
		max_q = np.amax(q_vals, axis=1).reshape([-1,1])
		targets = rewards + self.gamma * np.multiply((1-done_arr), max_q)
		__, loss_ = self.sess.run([self.optim_step, self.loss], feed_dict={self.state_placeholder: states, self.actions_placeholder:actions, self.targets:targets, self.learning_rate:lr})

class PolicyGradients:
	def __init__(self, env_):
		self.env_ = env_
		self.gamma = 0.95
		self.action_space = self.env_.env.action_space
		self.num_actions = self.action_space.n
		self.state_size = self.env_.env.observation_space.shape[0]
		self.hidden1_size = 64
		self.hidden2_size = 32
		self.moving_reward = None

		self.state_placeholder = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32, name='state')
		self.returns_placeholder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='returns')
		self.actions_placeholder = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32, name='actions')
		self.learning_rate = tf.placeholder(dtype=tf.float32, name='lr')

		self.hidden1 = tf.nn.relu(tf.layers.dense(self.state_placeholder, self.hidden1_size, kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None, name='hidden1'))
		self.hidden2 = tf.nn.relu(tf.layers.dense(self.hidden1, self.hidden2_size, kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None, name='hidden2'))
		self.action_logits = tf.layers.dense(self.hidden2, self.num_actions, kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None, name='action_logits')
		self.action_probs = tf.nn.softmax(self.action_logits, axis=1, name='action_probs')
		self.log_likelihood = tf.log(tf.clip_by_value(self.action_probs, 0.000001, 0.999999, name='clip'), name='log_likelihood')

		with tf.name_scope("loss_fn"):
			self.loss = -tf.reduce_mean(tf.multiply(self.returns_placeholder, tf.reshape(tf.reduce_sum(tf.multiply(self.log_likelihood, self.actions_placeholder), axis=1), [-1, 1])), axis=0)
			# self.loss = -tf.reduce_mean(tf.multiply(self.returns_placeholder, tf.reduce_sum(tf.multiply(self.log_likelihood, self.actions_placeholder), axis=1)))

		self.optim_step = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

		# For debugging
		with tf.variable_scope('action_logits', reuse=True):
			self.w_action_logits = tf.get_variable('kernel')
		with tf.variable_scope('hidden2', reuse=True):
			self.w_hidden = tf.get_variable('kernel')
		self.temp_grad = tf.gradients(self.loss, self.w_action_logits)
		
		self.sess = tf.Session()
		self.summary_op = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter("tensorboard/pg/", self.sess.graph)
		self.writer.close()
		self.sess.run(tf.global_variables_initializer())

	def train(self, env, episodes = 10, lr = 0.01, gamma = 0.95, update_steps = 10):
		self.gamma = gamma
		self.lr = lr
		all_moving_rewards=[]
		for episode in range(episodes):
			done = False
			obs = self.env_.reset()
			step = 0
			ep_start_time = time()
			self.buffer_ = []
			while not done:
				step+=1
				temp = {}
				action = self.action(obs)
				temp['state'] = obs
				temp['action'] = action
				obs, reward, done, info = self.env_.step(action)
				# if episode>1350:
				# 	self.env_.env.render()
				temp['reward'] = reward
				self.buffer_.append(temp)
			if self.moving_reward is None:
				self.moving_reward = float(sum(x['reward'] for x in self.buffer_))
			else:
				self.moving_reward = 0.99 * self.moving_reward + 0.01 * float(sum(x['reward'] for x in self.buffer_))
			all_moving_rewards.append(self.moving_reward)
			print("Episode:", episode, "Steps:", step, "reward:", self.moving_reward, "lr", self.lr, "Time:", time()-ep_start_time)
			self.update_net(self.lr)
		fig = plt.figure()
		plt.plot(range(episodes), all_moving_rewards)
		plt.xlabel("episode")
		plt.ylabel("total reward")
		fig.savefig("reward_curve_pg.png")

	def action(self, state):
		action_probs = self.sess.run(self.action_probs, feed_dict={self.state_placeholder:np.array(state).reshape(1, -1)})
		action = np.random.choice(list(range(self.num_actions)), p=action_probs[0])
		return action

	def update_net(self, lr = 0.001):
		states = np.array([x['state'] for x in self.buffer_])
		rewards = np.array([x['reward'] for x in self.buffer_])

		discounted_r = np.zeros_like(rewards)
		running_add = 0
		for t in reversed(range(0, rewards.size)):
			running_add = running_add * self.gamma + rewards[t]
			discounted_r[t] = running_add
		returns = discounted_r.reshape([-1, 1])

		actions = np.zeros([len(self.buffer_), self.num_actions])
		for i, x in enumerate(self.buffer_):
			temp_action = x['action']
			actions[i, temp_action] = 1

		__, loss_, temp_w = self.sess.run([self.optim_step, self.loss, self.w_hidden], feed_dict={self.state_placeholder: states, self.returns_placeholder:returns, self.actions_placeholder:actions, self.learning_rate:lr})

class RandomAgent:
	def __init__(self, env_):
		self.env_ = env_
		self.action_space = self.env_.env.action_space
		self.num_actions = self.action_space.n

	def train(self, env):
		pass

	def action(self, state = None):
		return self.action_space.sample()