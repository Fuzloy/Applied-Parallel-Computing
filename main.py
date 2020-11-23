import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.curandom import rand as curandom
from pycuda.driver import In, Out, Context
from pycuda.compiler import SourceModule
import random
import numpy as np
import time

N = 1000000  # Количество точек
BLOCK = (128, 1, 1)


def in_circle(x, y):
    if ((x ** 2 + y ** 2) <= 1):
        return True
    else:
        return False


def CPU():
    N_round = 0
    start = time.time()
    for _ in range(N):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)

        if in_circle(x, y):
            N_round += 1

    cpu_time = time.time() - start

    pi = 4 * N_round / N
    error = np.abs(np.pi - pi)
    return pi, round(cpu_time, 5), round(error, 5)


def GPU():
    N_round_GPU = gpuarray.zeros((1,), dtype=np.int32)
    N_round_GPU = N_round_GPU.get()
    grid_dim = (int(N / (128 ** 2)), 1)
    start = time.time()
    generated_x, generated_y = curandom((N,), dtype=np.double).get().astype(np.double), curandom((N,),
                                                                                                 dtype=np.double).get().astype(
        np.double)
    pi_calc = kernel.get_function("pi_calc")
    pi_calc(In(generated_x), In(generated_y), Out(N_round_GPU), np.int32(N), block=BLOCK, grid=grid_dim)
    Context.synchronize()

    pi = 4 * N_round_GPU[0] / N

    gpu_time = time.time() - start

    error = np.abs(np.pi - pi)
    return pi, round(gpu_time, 5), round(error, 5)


kernel = SourceModule(
    """
    __global__ void pi_calc(double *generated_x, double *generated_y, int *N_round_GPU, const int N){
        int counter = 0;
        int j = blockIdx.x * blockDim.x + threadIdx.x;

        for (int i = j; i < N; i += gridDim.x * blockDim.x) {
            if (generated_x[i]*generated_x[i] + generated_y[i]*generated_y[i] <= 1) {counter+=1;}
        }
        atomicAdd(N_round_GPU, counter);
    }
    """)

print(CPU())
print(GPU())