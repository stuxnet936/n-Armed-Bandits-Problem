import numpy as np 
from matplotlib import pyplot as plt 

class BanditMachine:
	def __init__(self, n_arms=10, epsilon=0, stdev=5.0, step_size=0.01):
		self.n_arms = n_arms
		self.epsilon = epsilon
		self.q_star = np.random.normal(loc=0, scale=stdev, size=n_arms)		#optimal q values
		self.q_est = np.zeros(n_arms)	#estimated q value of each action
		self.actions = np.arange(n_arms)
		self.action_count = np.zeros(n_arms)
		self.stdev = stdev
		self.averageReward = 0
		self.time = 0
		self.step_size = step_size
		#initalize estimated q for each action
		for i in range(0, n_arms):
			self.q_est[i] += self.getReward(i)

		self.optimal_action = np.argmax(self.q_star)		#index of optimal action

	def selectAction(self):
		if np.random.rand() > self.epsilon:
			return np.argmax(self.q_est)		#exploit action with max q value
		else:
			return np.random.choice(self.actions)	#explore randomly new actions

	def getReward(self, action):
		return np.random.normal(loc=self.q_star[action], scale=self.stdev)

	def step(self, action):
		reward = self.getReward(action)
		self.time += 1
		self.action_count[action] += 1
		# calculate average like this at each step or at the end
		self.averageReward = (1/self.time)*(reward + (self.time-1)*self.averageReward)
		#
		self.q_est[action] += self.step_size*(reward - self.q_est[action])
		return reward


def runAgent(nCycles, time, bandits):
	#lists for values of different bandit machines
	bestActionCounts = [np.zeros(time) for _ in range(0, nCycles)]
	averageRewards = [np.zeros(time) for _ in range(0, nCycles)]
	for k, bandit in enumerate(bandits):	#different bandit machines with different policies
		for i in range(0, nCycles):
			for t in range(0, time):
				action = bandit[i].selectAction()
				reward = bandit[i].step(action)
				averageRewards[k][t] += reward
				if action == bandit[i].optimal_action:
					bestActionCounts[k][t] += 1
		averageRewards[k] /= nCycles
		bestActionCounts[k] /= nCycles
		averageRewards[k] = np.cumsum(averageRewards[k])/(np.arange(len(averageRewards[k]))+1)
		bestActionCounts[k] = np.cumsum(bestActionCounts[k])/(np.arange(len(bestActionCounts[k]))+1)
	return averageRewards, bestActionCounts


def epsilonGreedy(nCycles, time):
	epsilons = [0, 0.01, 0.1]
	bandits = []		# list of bandit machines with different policies
	for k, eps in enumerate(epsilons):
		bandits.append([BanditMachine(epsilon=eps) for _ in range(nCycles)])
	averageRewards, bestActionCounts = runAgent(nCycles, time, bandits)

	global figureIndex
	# figure 1 - Average rewards
	plt.figure(figureIndex)
	figureIndex += 1
	colors = ['red', 'blue', 'green', 'orange', 'yellow']
	k=0
	for rewards, eps in zip(averageRewards, epsilons):
		col = colors[k]
		k += 1
		steps = np.arange(len(rewards))
		plt.plot(steps+10, rewards, color=col, lw=1.5, label='epsilon = {0}'.format(eps))
	plt.legend(loc='best')
	plt.grid()
	plt.xlabel('# of steps')
	plt.ylabel('Average reward over past steps')
	# figure 2 - optimal actions
	plt.figure(figureIndex)
	figureIndex += 1
	k=0
	for actions, eps in zip(bestActionCounts, epsilons):
		col = colors[k]
		k += 1
		steps = np.arange(len(actions))
		plt.plot(steps+10, actions, color=col, lw=1.5, label='epsilon = {0}'.format(eps))
	plt.legend(loc='best')
	plt.grid()
	plt.xlabel('# of steps')
	plt.ylabel('% optimal actions')
	
	plt.show()

figureIndex = 0

if __name__ == "__main__":
	epsilonGreedy(2000, 1000)