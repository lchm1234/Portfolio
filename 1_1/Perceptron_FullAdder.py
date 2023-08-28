import numpy as np

# NAND 게이트
def NAND(x1, x2):
    # 0 또는 1의 입력값
    x = np.array([x1, x2])
    # 가중치
    w = np.array([-0.5, -0.5])
    # 바이어스
    b = 0.7
    # 두 입력값 중 하나라도 0이면 1을 반환
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
# AND 게이트
def AND(x1, x2):
    # 가중치 설정
    w1, w2, theta = 0.5, 0.5, 0.7
    # 두 입력값 모두 1이면 1을 반환
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

# OR 게이트
def OR(x1, x2):
    # 0 또는 1의 입력값
    x = np.array([x1, x2])
    # 가중치
    w = np.array([0.5, 0.5])
    # 바이어스
    b = -0.2
    # 두 입력값 중 하나라도 1이면 1을 반환
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
# XOR 게이트
def XOR(x1, x2):
    # XOR 게이트는 하나의 층으로 구현할 수 없음
    # 단일 퍼셉트론의 한계
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

# 전가산기
def Full_Adder(x1, x2, x3):
    s1 = XOR(x1, x2)
    s2 = AND(x1, x2)
    s3 = XOR(s1, x3)
    s4 = AND(s1, x3)
    s5 = OR(s4, s2)
    return '(%d, %d)' % (s3, s5)

# 결과값 출력
print("(0, 0, 0) => " + Full_Adder(0,0,0))
print("(0, 0, 1) => " + Full_Adder(0,0,1))
print("(0, 1, 0) => " + Full_Adder(0,1,0))
print("(0, 1, 1) => " + Full_Adder(0,1,1))
print("(1, 0, 0) => " + Full_Adder(1,0,0))
print("(1, 0, 1) => " + Full_Adder(1,0,1))
print("(1, 1, 0) => " + Full_Adder(1,1,0))
print("(1, 1, 1) => " + Full_Adder(1,1,1))