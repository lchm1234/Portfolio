import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

#시그모이드 함수 결과 출력
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# #relu 함수 결과 출력
# x = np.arange(-5.0, 5.0, 0.1)
# y = relu(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()