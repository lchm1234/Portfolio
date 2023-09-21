import numpy as np
import matplotlib.pyplot as plt

# 0 ~ 1 사이값을 반환하는 시그모이드 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 입력값이 음수이면 0을 반환하고, 양수이면 그대로 반환하는 relu 함수 정의
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