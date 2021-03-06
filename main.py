import numpy as np
from environment import Environment
from algorithms import RandomAgent, PolicyGradients, DQN


def test(agent, env, num_runs = 10, render=False):
	print("Testing")
	all_runs_rew = []
	for run in range(num_runs):
		obs = env.reset()
		done = False
		total_reward = 0
		for t in range(100):
			if done:
				break
			obs, reward, done, info = env.step(agent.action(obs))
			total_reward+=reward
		all_runs_rew.append(total_reward)
	env.close()
	print("mean reward:", float(sum(all_runs_rew))/len(all_runs_rew))
	return all_runs_rew

if __name__=='__main__':
	# env_ = Environment(env_name="LunarLander-v2", render = False)
	# env_ = Environment(env_name="MountainCar-v0", render = False)
	env_ = Environment(env_name="CartPole-v1", render = False)
	# agent = PolicyGradients(env_)
	agent = DQN(env_, buffer_size = 500000, batch_size = 512)
	agent.train(env_, episodes=6000, lr=0.01, gamma=1)

	print('Training done')
	test(agent, env_)