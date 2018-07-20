""" 
Trying at solving Frozen Lake problem by policy iteratiopon

Author: Shivansh Mundra
"""
import numpy as np 
import gym
# from gym import wrappers


def run_episode(env, policy, gamma = 1.0, render = False):
	"""
	@brief      { to run a episode in env }
	
	@param      env     The environment
	@param      policy  The policy
	@param      gamma   The gamma
	@param      render  bool
	
	@return     { total delayed rewardd }
	"""
	state = env.reset()
	# state: current state
	step = 0
	total_reward = 0
	while True:
		if render:
			env.render()
		state, reward, done, _ = env.step(int(policy[state]))
		total_reward += ((gamma**step)*reward)
		step += 1
		if done:
			break

	return total_reward

def evaluate_policy(policy, env, gamma= 1.0, n = 100, render = False):
	"""
	@brief      evaluate how good a policy by running n times	
	@param      n       iterations of episodes

	@return     score of a policy
	"""
	reward = []
	for x in range(1,n):
		rew = run_episode(env, policy, gamma)
		reward.append(rew)
	return np.mean(reward)

def making_policy(env, val, gamma = 1.0):
	"""
	@brief      return policy according to value function
	
	@param      val    The value
	@param      gamma  The gamma
	
	@return     { description_of_the_return_value }
	"""
	policy = np.zeros(env.observation_space.n)
	for s in xrange(env.observation_space.n):
		q_sa = np.zeros(env.action_space.n)
		for a in xrange(env.action_space.n):
			next_q = env.P[s][a]
			for id in next_q:
				pr, nex_sta, rew, done = id
				q_sa[a] += pr*(rew + gamma*val[nex_sta])

		policy[s] = np.argmax(q_sa)
	return policy

def value_iteration(env, gamma = 1.0):
	"""
	@brief      value iteration according to bellman equation
	
	@param      env    The environment
	@param      gamma  The gamma
	
	@return    converged value function
	"""
	eps = 1e-20
	v = np.zeros(env.observation_space.n)
	error = np.inf

	max_iter = 10000
	for i in range(max_iter):
		prev_v = np.copy(v)
		for s in range(env.observation_space.n):
			q_sa = np.zeros(env.action_space.n)
			for a in range(env.action_space.n):
				next_q = env.P[s][a]
				for id in next_q:
					pr, nex_sta, rew, done = id
					q_sa[a] += pr*(rew + gamma*prev_v[nex_sta])
			v[s] = np.max(q_sa)

		if np.sum(np.abs(prev_v - v)) < eps:
			print('policy converged at %d iterations\n' %(i+1))
			break
	return v

if __name__ == '__main__':
	env_name = 'FrozenLake8x8-v0'
	gamma = 1.0
	env = gym.make(env_name)
	print(env.observation_space.n)
	optimal_v = value_iteration(env, gamma)
	policy = making_policy(env, optimal_v, gamma)
	policy_score = evaluate_policy(policy, env, gamma)
	print('Policy average score = ', policy_score)





