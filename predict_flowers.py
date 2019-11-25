'''example from a giant_neural_network youtube channel'''

import numpy as np 
import matplotlib.pyplot as plt 


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	sig = sigmoid(x)
	return sig * (1 - sig)

def weight_me(point):
	return point[0] * w1 + point[1] * w2 + b

def train():
	w1, w2, b = np.random.randn(), np.random.randn(), np.random.randn()
	for iteration in range(iterations):
		rand_index = np.random.randint(len(dataset))
		point = dataset[rand_index]

		weighted = point[0] * w1 + point[1] * w2 + b 	#weight_me
		probability = sigmoid(weighted)
		error = np.square(probability - point[2])

		d_error_prediction = 2 * (probability - point[2])
		dpred_dsig = derivative_sigmoid(weighted) * d_error_prediction

		derror_w1 = dpred_dsig * point[0]
		derror_w2 = dpred_dsig * point[1]
		derror_b = dpred_dsig

		w1 = w1 - learning_rate * derror_w1
		w2 = w2 - learning_rate * derror_w2
		b = b - learning_rate * derror_b
	return w1,w2,b

def generate_arrays(x,y, limit = 20):
	xs = np.linspace(0, x, limit)
	ys = np.linspace(0, y, limit)
	np.random.shuffle(xs)
	np.random.shuffle(ys)
	return np.dstack((xs, ys))


if __name__ == '__main__':
	dataset = [ [3, 1.5, 1],
       			[2, 1, 0],
        		[4, 1.5, 1],
        		[3, 1, 0],
        		[3.5, 0.5, 1],
        		[2, 0.5, 0],
        		[5.5, 1, 1],
        		[1, 1, 0] ]
	new_dataset  = generate_arrays(6,3)[0]

	learning_rate = 0.2
	iterations = 10**5
	w1, w2, b = train()

	plt.axis([0, 6.5, 0, 3.2])
	plt.grid()
	for point in dataset:
		plt.scatter(point[0], point[1], c = 'r' if point[2] else 'b', marker = '*')

	for point in new_dataset:
		prediction = sigmoid(weight_me(point))
		percentage = 100*prediction if prediction > 0.5 else (1-prediction)*100
		print(f'point{point} {"blue" if prediction < 0.5 else "red"} with {percentage} %')
		plt.scatter(point[0], point[1], c = 'b' if prediction < 0.5 else 'r')
	plt.show()
