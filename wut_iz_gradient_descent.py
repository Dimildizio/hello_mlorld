import random
from matplotlib import pyplot as plt

w1 = random.random()
w2 = random.random()
learning_rate = 0.05

house_size = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]
house_price = [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]

standartize = lambda data: [round((dp - min(data)) / (max(data) - min(data)), 3) for dp in data]

dataX = standartize(house_size)
dataY = standartize(house_price)

total_sse, total_w1, total_w2, total_dssedw1, total_dssedw2 = [],[],[], [],[]
difference = 1

while True:
    sse, dssedw1, dssedw2 = 0,0,0
    for i in range(len(dataX)):
        
        YP = w1 + (w2*dataX[i])
        sse += 1/2 * (dataY[i] - YP)**2
        dssedw1 += - (dataY[i] - YP)
        dssedw2 += - (dataY[i] - YP) * dataX[i]
        
    w1 = w1 - (dssedw1 * learning_rate)
    w2 = w2 - (dssedw2 * learning_rate)
    
    total_w1.append(w1)
    total_w2.append(w2)
    total_sse.append(sse)
    total_dssedw1.append(dssedw1)
    total_dssedw2.append(dssedw2)
    
    if abs(difference - sse) < 0.00001: break
    else: difference = sse


iterations = [x for x in range(len(total_sse))]
print(total_sse[-2], total_sse[-1])
ax1 = plt.subplot(313)
ax1.plot(iterations, total_sse)
ax1.set_title('Sum of squared errors')
ax1.set(xlabel="Iterations")

ax2 = plt.subplot(321)         
ax2.plot(iterations, total_w1)
ax2.set_title('Weight 1')

ax3 = plt.subplot(322)
ax3.plot(iterations, total_w2)
ax3.set_title('Weight 2')

ax3 = plt.subplot(323)
ax3.plot(iterations, total_dssedw1)
ax3.set_title('p.derivative weight 1')

ax3 = plt.subplot(324)
ax3.plot(iterations, total_dssedw2)
ax3.set_title('p.derivative weight 2')


plt.tight_layout()
plt.show()
