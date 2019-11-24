'''simple nn'''

import numpy as np 


class NeuralNetwork:
	def __init__(self):
		#initializing random weights from 0 to 1
		self.synaptic_weights = 2 * np.random.random((3,1)) - 1 

	def sigmoid(self, x):
		#activation function 1 / (1 + (e to the power of -n for n in x))
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivative(self, x):
		#gradient descent of sigmoid to minimize loss
		return x * (1 - x)

	def train(self, t_in, t_out, iterations = 10000):
		#train with backpropogation 
		for iteration in range(iterations):
			output = self.new_output(t_in)
			error = t_out - output
			adjust = np.dot(t_in.T, error * self.sigmoid_derivative(output))
			self.synaptic_weights += adjust


	def new_output(self, t_in):
		#activate and get the result
		return self.sigmoid(t_in @ self.synaptic_weights)


if __name__ == '__main__':
	training_inputs = np.array([[0,0,1],
								[1,1,1],
								[1,0,1],
								[0,1,1]])
	training_outputs = np.array([[0,1,1,0]]).T

	neural_network = NeuralNetwork()
	print('wights before:\n', neural_network.synaptic_weights)
	neural_network.train(training_inputs, training_outputs)
	print('weights after:\n', neural_network.synaptic_weights)

	for num1 in range(2):
		for num2 in range(2):
			for num3 in range(2):
				print(f'probability for [{num1, num2, num3}]: {neural_network.new_output(np.array([num1,num2,num3]))}')


