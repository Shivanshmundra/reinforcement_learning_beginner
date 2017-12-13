import numpy as np 
import tensorflow as tf 
import tensorflow.contrib.slim as slim

class Contextual_bandits(object):
	"""docstring for Contextual_bandits"""
	def __init__(self):
		self.state = 0

		self.bandits = np.array([[1.0, 4.5, -2.5, -3.6], [0.6, 6.3, -1.6, -3.4],
			[-2.0, 4.5, -2.7, -1.9], [2.4, 0.1, -0.5, -0.9]])

		self.num_states = self.bandits.shape[0]
		self.len_bandits = self.bandits.shape[1]

	def get_bandit_state(self):

		self.state = np.random.randint(0, self.num_states)

		return self.state

	def pull_a_bandit(self, chosen_action):
		random = np.random.randn(1)

		bandit = self.bandits[self.state, chosen_action]

		if bandit>random:
			return 1
			#positive reward
		else:
			return -1
			#negative reward


class My_agent(object):
	"""docstring for My_agent"""
	def __init__(self, learning_rate, state_size, action_size):
		self.state_current = tf.placeholder(shape = [1], dtype = tf.int32)

		state_in_onehot = slim.one_hot_encoding(self.state_current, state_size)

		output = slim.fully_connected(state_in_onehot, action_size, biases_initializer=None, activation_fn=tf.nn.sigmoid,
			weights_initializer=tf.ones_initializer())

		self.output = tf.reshape(output, [-1])
		self.chosen_action = tf.argmax(self.output, 0)


		self.reward_holder = tf.placeholder(shape = [1], dtype = tf.float32)#stores reward from pull_a_bandit

		self.action_holder = tf.placeholder(shape = [1], dtype = tf.int32)#stores action selection		

		self.responsible_weight = tf.slice(self.output,self.action_holder,[1])#stores current value of chosen action 

		self.loss = (-tf.log(self.responsible_weight)*self.reward_holder)

		optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)


		self.update = optimizer.minimize(self.loss)


tf.reset_default_graph()

c_bandit = Contextual_bandits()
my_agent = My_agent(learning_rate=0.001, state_size=c_bandit.num_states, action_size=c_bandit.len_bandits)

weights = tf.trainable_variables()[0]#these are variables that can be updated using gradient_descent
									#just used in model as a variable to be updated

total_num_test = 100000

total_reward = np.zeros([c_bandit.num_states, c_bandit.len_bandits])

e = 0.2

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	i = 0

	while i < total_num_test:
		current_state = c_bandit.get_bandit_state()

		e_random = np.random.randn(1)#to distribute action selection according to epsilon greedy approach

		if e_random>e:
			action = sess.run(my_agent.chosen_action, feed_dict={my_agent.state_current:[current_state]})
		else:
			action = np.random.randint(c_bandit.len_bandits)


		reward = c_bandit.pull_a_bandit(action)

		feed_dict={my_agent.reward_holder:[reward], my_agent.action_holder:[action], my_agent.state_current:[current_state]}

		_,curr_weight = sess.run([my_agent.update, weights], feed_dict=feed_dict)

		total_reward[current_state, action] += reward

		i+=1
		if i%500==0:
			print "Mean reward for each of the " + str(c_bandit.num_states) + " bandits: " + str(np.mean(total_reward,axis=1))
    	

	for x in xrange(c_bandit.num_states):
		#print "The agent thinks action "+ str(np.argmax(curr_weight[x])+1)+ " for bandit "+str(x+1)+" is best"
		print ((curr_weight[x]))
		if np.argmax(curr_weight[x])==(np.argmax(c_bandit.bandits[x])):
			print "....you are right"
		else:
			print"...sorry you are wrong"

			

		


		
		

		