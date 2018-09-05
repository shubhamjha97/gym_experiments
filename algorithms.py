import numpy as np
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt

class PolicyGradients:
	def __init__(self, env_):
		self.env_ = env_
		self.gamma = 0.95
		self.buffer_ = []
		self.action_space = self.env_.env.action_space
		self.num_actions = self.action_space.n
		self.state_size = self.env_.env.observation_space.shape[0]
		self.hidden1_size = 10
		self.hidden2_size = 10
		self.moving_reward = None

		self.state_placeholder = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32, name='state')
		self.returns_placeholder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='returns')
		self.actions_placeholder = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32, name='actions')
		self.learning_rate = tf.placeholder(dtype=tf.float32, name='lr')

		self.hidden1 = tf.nn.relu(tf.layers.dense(self.state_placeholder, self.hidden1_size, kernel_initializer = tf.contrib.layers.xavier_initializer(seed=1), activation=None, name='hidden1'))
		self.hidden2 = tf.nn.relu(tf.layers.dense(self.hidden1, self.hidden2_size, kernel_initializer = tf.contrib.layers.xavier_initializer(seed=1), activation=None, name='hidden2'))
		self.action_logits = tf.layers.dense(self.hidden2, self.num_actions, kernel_initializer = tf.contrib.layers.xavier_initializer(seed=1), activation=None, name='action_logits')
		self.action_probs = tf.nn.softmax(self.action_logits, axis=1, name='action_probs')
		self.log_likelihood = tf.log(tf.clip_by_value(self.action_probs, 0.000001, 0.999999, name='clip'), name='log_likelihood')

		with tf.name_scope("loss_fn"):
			self.loss = -tf.reduce_mean(tf.multiply(self.returns_placeholder, tf.reduce_sum(tf.multiply(self.log_likelihood, self.actions_placeholder), axis=1)))

		# with tf.name_scope('loss'):
		# 	self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.action_logits, labels=self.actions_placeholder)
		# 	self.loss = tf.reduce_mean(self.neg_log_prob * self.returns_placeholder)  # reward guided loss

		self.optim_step = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

		# For debugging
		with tf.variable_scope('hidden2', reuse=True):
			self.w = tf.get_variable('kernel')
		self.temp_grad = tf.gradients(self.loss, self.w)

		self.sess = tf.Session()
		self.summary_op = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter("tensorboard/", self.sess.graph)
		self.writer.close()
		self.sess.run(tf.global_variables_initializer())

	def train(self, env, episodes = 10, lr = 0.01, gamma = 0.95):
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
				obs, reward, done, info = self.env_.step(self.action(obs)) #(episode%1==0))
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
		fig.savefig("reward_curve.png")

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

		__, loss_= self.sess.run([self.optim_step, self.loss], feed_dict={self.state_placeholder: states, self.returns_placeholder:returns, self.actions_placeholder:actions,self.learning_rate:lr})

class RandomAgent:
	def __init__(self, env_):
		self.env_ = env_
		self.action_space = self.env_.env.action_space
		self.num_actions = self.action_space.n

	def train(self, env):
		pass

	def action(self, state = None):
		return self.action_space.sample()