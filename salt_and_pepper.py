import numpy as np
from pycuda.driver import In, Out, Context
from pycuda.compiler import SourceModule
import time
from img_import import save_image, load_image

BLOCK_SIZE = 32
BLOCK = (BLOCK_SIZE, BLOCK_SIZE, 1)
FILTER_SIZE = 5
ARRAY_SIZE = FILTER_SIZE ** 2
OFFSET = FILTER_SIZE // 2

def cpu_filtration(pixels, width, height):
    start = time.time()
    new = np.zeros_like(pixels)

    for i in range(height):
        for j in range(width):
            grid = np.zeros(ARRAY_SIZE)
            for k in range(FILTER_SIZE):
                x = max(0, min(i + k - OFFSET, height - 1))
                index = k * FILTER_SIZE
                for l in range(FILTER_SIZE):
                    y = max(0, min(j + l - OFFSET, width - 1))
                    grid[index + l] = pixels[x, y]
            grid.sort()
            new[i, j] = grid[ARRAY_SIZE // 2]

    cpu_time = time.time() - start
    save_image(new, "CPU img")
    return cpu_time

def gpu_filtration(pixels, width, height):
    start = time.time()
    new = np.zeros_like(pixels)

    size = np.array([width, height])
    grid_dim = (width // BLOCK_SIZE, height // BLOCK_SIZE)
    median_filter = kernel.get_function("median_filter")
    median_filter(In(pixels), Out(new), In(size), block=BLOCK, grid=grid_dim)
    Context.synchronize()

    gpu_time = time.time() - start
    save_image(new, "GPU img")
    return gpu_time

kernel = SourceModule(
    """
    __global__ void median_filter(unsigned char* pixels, unsigned char* filtered, int* size){
        const int blockSize = %(BLOCK_SIZE)s;
        const int arraySize = %(ARRAY_SIZE)s;
        const int filterSize = %(FILTER_SIZE)s;
        const int offset = %(OFFSET)s;
        int width = size[0];

        int j = blockIdx.x * blockDim.x + threadIdx.x;
	    int i = blockIdx.y * blockDim.y + threadIdx.y;

	    int x, y, index;

        __shared__ int local[blockSize][blockSize];
        int arr[arraySize];

        local[threadIdx.y][threadIdx.x] = pixels[i * width + j];
        __syncthreads ();

        for (int k = 0; k < filterSize; k++){
            x = max(0, min(threadIdx.y + k - offset, blockSize - 1));
            for (int l = 0; l < filterSize; l++){
                index = k * filterSize + l;
                y = max(0, min(threadIdx.x + l - offset, blockSize - 1));
                arr[index] = local[x][y];
            }
        }
        __syncthreads ();

        for (int k = 0; k < arraySize; k++){
            for (int l = k + 1; l < arraySize; l++){
                if (arr[k] > arr[l]){
                    unsigned char temp = arr[k];
                    arr[k] = arr[l];
                    arr[l] = temp;
                }
            }
        }

        filtered[i * width + j] = arr[int(arraySize / 2)];
    }
    """ % {
        'BLOCK_SIZE': BLOCK_SIZE,
        'ARRAY_SIZE': ARRAY_SIZE,
        'OFFSET': OFFSET,
        'FILTER_SIZE': FILTER_SIZE
    }
)

def main():

    file = "512x512.bmp"
    pixels, width, height = load_image(file)
    c_time = cpu_filtration(pixels, width, height)
    g_time = gpu_filtration(pixels, width, height)

    acceleration = c_time / g_time

    print("Время на CPU: ", c_time)
    print("Время на GPU: ", g_time)
    print("Ускорение: ", acceleration)