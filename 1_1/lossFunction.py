import numpy as np

# 교차 엔트로피 오차
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# 2번방을 정답으로 지정
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# 가중치 초기화
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))

# 가중치 초기화
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))

# 수치 미분
def numerical_diff(f, x):
    h = 1e04
    return (f(x+h) - f(x-h)) / (2*h)

def function_tmp1(x0):
    return x0*x0 + 4.0 ** 2.0

print(numerical_diff(function_tmp1, 3.0))

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

print(numerical_diff(function_tmp2, 4.0))