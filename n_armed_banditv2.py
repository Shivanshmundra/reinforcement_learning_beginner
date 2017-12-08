import numpy as np 

bandits = [0.4, 0.8, -0.4, 1.5, 1.7]
len_bandits = len(bandits)
num_test = 2000

Q_value = {}
Number_times = {}

def pullbandit(bandit):
	random = np.random.randn(1)
	if bandit>random:
		return 1
		#positive reward
	else:
		return -1
		#negative reward

for x in xrange(0,len_bandits):
	Q_value[x] = np.random.randn(1)
	Number_times[x] = 1

for x in xrange(1,num_test):
	key = np.random.randint(1,len_bandits)
	reward = pullbandit(bandits[key-1])

	Q_value[key] += ((reward - Q_value[key])/Number_times[key])

	Number_times[key] += 1

print("Max reward holder bandit is: ")
print(bandits[max(Q_value, key = Q_value.get)])
print(Q_value)
