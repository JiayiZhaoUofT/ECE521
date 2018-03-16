import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

y_head = x = np.arange(0., 1, 0.01)

y = 0

squared_error = np.square(y_head - 0)
cross_entropy_loss = -1 * np.log(1-y_head)

#print(squared_error)

plt.xlabel('y_head')
plt.plot(y_head, squared_error, label = 'squared error')
plt.plot(y_head, cross_entropy_loss, label = 'cross-entropy loss')
plt.legend(loc = 'upper center')
plt.show()
