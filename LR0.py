import numpy as np
import cupy as cp
import time


def matrix_creation_cpu(n):
    a = np.ones((n, n), 'f')
    b = np.ones((n, n), 'f')
    return a, b

def matrix_creation_gpu(n):
    a = cp.ones((n, n), 'f')
    b = cp.ones((n, n), 'f')
    return a, b

def dot_computation_cpu(a,b):
    # Умножение матриц с помощью numpy
    s = time.time()
    dot_product_cpu = np.dot(a, b)
    return dot_product_cpu, time.time() - s

def dot_computation_gpu(a,b):
    # Умножение матриц с помощью cupy
    s = time.time()
    dot_computation_gpu = cp.dot(a, b)
    return dot_computation_gpu, time.time() - s

n = 100

#Инициализация матриц

#CPU
ACPU, BCPU = matrix_creation_cpu(n)

#GPU
AGPU, BGPU = matrix_creation_gpu(n)

#Вычисления

CCPU, ctime1 = dot_computation_cpu(ACPU, BCPU)

CGPU, ctime2 = dot_computation_gpu(AGPU, BGPU)

acseleration = ctime2/ctime1

print(ctime1, ctime2, acseleration)