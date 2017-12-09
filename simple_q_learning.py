import numpy as np 
import gym


env = gym.make('FrozenLake-v0')

Q_table = np.zeros((env.observation_space.n, env.action_space.n))

total_episodes = 2000

reward_list = []
goal_reached = False
learning_rate = 0.75
discount_f = 0.95
curr_reward = 0


for i in xrange(1,total_episodes):
	curr_reward = 0
	curr_state = env.reset()
	goal_reached = False
	j = 0
	while j < 99:
		j+=1
		action = np.argmax(Q_table[curr_state,:] + (np.random.randn(1,env.action_space.n))*(1./(i+1)))

		next_state, reward, goal_reached,_ = env.step(action)

		Q_table[curr_state,action] += learning_rate*(reward + discount_f*(np.argmax(Q_table[next_state,:])) - Q_table[curr_state,action]) 

		curr_reward += reward

		curr_state = next_state

		if goal_reached==True:
			break
		

	reward_list.append(curr_reward)

print " Average Score over time is " +str(np.sum(reward_list)/(total_episodes))

print "Q_table is "
print (Q_table)
