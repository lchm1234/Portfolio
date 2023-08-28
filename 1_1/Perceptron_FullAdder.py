import numpy as np

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

def Full_Adder(x1, x2, x3):
    s1 = XOR(x1, x2)
    s2 = AND(x1, x2)
    s3 = XOR(s1, x3)
    s4 = AND(s1, x3)
    s5 = OR(s4, s2)
    return '(%d, %d)' % (s3, s5)

print("(0, 0, 0) => " + Full_Adder(0,0,0))
print("(0, 0, 1) => " + Full_Adder(0,0,1))
print("(0, 1, 0) => " + Full_Adder(0,1,0))
print("(0, 1, 1) => " + Full_Adder(0,1,1))
print("(1, 0, 0) => " + Full_Adder(1,0,0))
print("(1, 0, 1) => " + Full_Adder(1,0,1))
print("(1, 1, 0) => " + Full_Adder(1,1,0))
print("(1, 1, 1) => " + Full_Adder(1,1,1))