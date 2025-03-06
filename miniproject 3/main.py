import time
import warnings
import mandelbrot_tools
import numpy
import dask.array as da
warnings.filterwarnings("ignore")

def vectorized(input_arr, iterations=100):
    mandelbrot = input_arr.copy()
    st = time.time()
    member = mandelbrot_tools.stable_vectorized(mandelbrot, 2, iterations)
    end = time.time()
    elapsed_time = end - st
    print(f"{elapsed_time=} FOR 1 process vectorized numpy")
    mandelbrot_tools.plot_mandelbrot(member)
    
def dask_vectorized(input_arr, chunk_size="384KiB", iterations=100):
    dask_c = da.from_array(input_arr, chunks=chunk_size)
    st = time.time()
    dask_z = da.map_blocks(mandelbrot_tools.stable_vectorized, dask_c, 2, iterations)
    z = dask_z.compute()
    end = time.time()
    elapsed_time = end - st
    print(f"Elapsed time = {elapsed_time}, for 1 process vectorized dask with, {chunk_size = }")
    mandelbrot_tools.plot_mandelbrot(z)
    


if __name__ == '__main__':
    mandelbrot_arr = mandelbrot_tools.generate_arrays(-2, 1, -1.5, 1.5, 2000)
    mandelbrot_arr_256 = mandelbrot_tools.generate_arrays_256(-2, 1, -1.5, 1.5, 2000)
    mandelbrot_arr_128 = mandelbrot_tools.generate_arrays_128(-2, 1, -1.5, 1.5, 2000)
    mandelbrot_arr_64 = mandelbrot_tools.generate_arrays_64(-2, 1, -1.5, 1.5, 2000)
    mandelbrot_arr_32 = mandelbrot_tools.generate_arrays_32(-2, 1, -1.5, 1.5, 2000)
    mandelbrot_arr_16 = mandelbrot_tools.generate_arrays_16(-2, 1, -1.5, 1.5, 2000)
    iterations = 500
    
    vectorized(mandelbrot_arr, iterations)
    dask_vectorized(mandelbrot_arr, "384KiB", iterations)
    dask_vectorized(mandelbrot_arr, "3MiB", iterations)
    dask_vectorized(mandelbrot_arr, "32MiB", iterations)
    
    vectorized(mandelbrot_arr_256, iterations)
    dask_vectorized(mandelbrot_arr_256, "384KiB", iterations)
    dask_vectorized(mandelbrot_arr_256, "3MiB", iterations)
    dask_vectorized(mandelbrot_arr_256, "32MiB", iterations)
    
    vectorized(mandelbrot_arr_128, iterations)
    dask_vectorized(mandelbrot_arr_128, "384KiB", iterations)
    dask_vectorized(mandelbrot_arr_128, "3MiB", iterations)
    dask_vectorized(mandelbrot_arr_128, "32MiB", iterations)
    
    vectorized(mandelbrot_arr_64, iterations)
    dask_vectorized(mandelbrot_arr_64, "384KiB", iterations)
    dask_vectorized(mandelbrot_arr_64, "3MiB", iterations)
    dask_vectorized(mandelbrot_arr_64, "32MiB", iterations)
    
    vectorized(mandelbrot_arr_32, iterations)
    dask_vectorized(mandelbrot_arr_32, "384KiB", iterations)
    dask_vectorized(mandelbrot_arr_32, "3MiB", iterations)
    dask_vectorized(mandelbrot_arr_32, "32MiB", iterations)
    
    vectorized(mandelbrot_arr_16, iterations)
    dask_vectorized(mandelbrot_arr_16, "384KiB", iterations)
    dask_vectorized(mandelbrot_arr_16, "3MiB", iterations)
    dask_vectorized(mandelbrot_arr_16, "32MiB", iterations)