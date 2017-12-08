import numpy as np 
import tensorflow as tf 


bandits = [0.5, 1.2, -0.6, 1.7, 0.0]
num_bandits = len(bandits)

def pullbandit(bandit):
	random = np.random.randn(1)
	if bandit>random:
		return 1
		#postive reward
	else: 
		#negative reward
		return -1

tf.reset_default_graph()

values = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(values,0)

#here value is the current value of respective action


reward_holder = tf.placeholder(shape = [1], dtype = tf.float32)
action_holder = tf.placeholder(shape = [1], dtype = tf.int32)
responsible_value = tf.slice(values, action_holder, [1])
#here responsible weight is current responsible value for that action
# in action holder

loss = -(tf.log(responsible_value)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
update = optimizer.minimize(loss)


total_episodes = 1000 #no. of total steps
total_reward = np.zeros(num_bandits)

e = 0.1 #this is epsilon

init = tf.initialize_all_variables()

#launching tf graph

with tf.Session() as sess:
	sess.run(init)
	i = 0
	while i < total_episodes:

	 	#choosing either a random action or exploiting 
	 	#based on value of epsilon
	 	if np.random.randn(1) < e:
	 		action = np.random.randint(num_bandits)
	 	else:
	 		action = sess.run(chosen_action)

	 	reward = pullbandit(bandits[action])
	 	#updating

	 	_,resp,ww = sess.run([update, responsible_value, values],
	 		 feed_dict = {reward_holder:[reward], action_holder:[action] })

	 	total_reward[action] += reward
	 	if i % 50 == 0:
	 		print "Running reward for the " + str(num_bandits) +" bandits are" + str(total_reward)	 
	 	i+=1

if np.argmax(ww) == np.argmax(np.array(bandits)):
	print " ...and it was right "
else:
	print "...sorry it was wrong"