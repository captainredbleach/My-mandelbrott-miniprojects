import time
import warnings
import mandelbrot_tools
import numpy as np
import dask.array as da
import pyopencl as cl
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def mandelbrot_OpenCL(input_arr, iterations=100):
    """
    Runs the vectorized mandelbrot calculation with opencl on the GPU

    Parameters
    ----------
        input_arr (ndarray): The input array
        iterations (int, ndarray): The maximum number of iterations. Defaults to 100.

    Returns
    ----------
        float: The time it took to calculate the mandelbrott set
    """
    platform = cl.get_platforms()
    gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=gpu_devices)
    member, elapsed_time = mandelbrot_tools.opencl_vectorized(input_arr, iterations, ctx)
    mandelbrot_tools.plot_mandelbrot(member)
    return elapsed_time
    
    
def vectorized(input_arr, iterations=100):
    """
    Runs the vectorized mandelbrot calculation

    Parameter
    ----------
        input_arr (ndarray): The input array
        iterations (int, optional): The maximum number of iterations. Defaults to 100.

    Returns
    ----------
        float: The time it took to calculate the mandelbrott set
    """
    mandelbrot = input_arr.copy()
    st = time.time()
    member = mandelbrot_tools.stable_vectorized(mandelbrot, 2, iterations)
    end = time.time()
    elapsed_time = end - st
    mandelbrot_tools.plot_mandelbrot(member)
    return elapsed_time
    

def dask_vectorized(input_arr, chunk_size="384KiB", iterations=100):
    """
    Runs the vectorized mandelbrot calculation with dask

    Parameters
    ----------
        input_arr (ndarray): The input array
        chunk_size (str, optional): The chunk size dask creates. Defaults to "384KiB".
        iterations (int, optional): The maximum number of iterations. Defaults to 100.

    Returns
    ----------
        float: The time it took to calculate the mandelbrott set
    """
    dask_c = da.from_array(input_arr, chunks=chunk_size)
    dask_z = da.map_blocks(mandelbrot_tools.stable_vectorized, dask_c, 2, iterations)
    st = time.time()
    z = dask_z.compute()
    end = time.time()
    elapsed_time = end - st
    mandelbrot_tools.plot_mandelbrot(z)
    return elapsed_time




if __name__ == '__main__':
    iterations = 1000
    Array_size = [154,433,1414,2000,4000]#["384KiB", "3MiB", "32MiB", "64MiB", "256MiB"]
    vectorized_times = []
    dask_vectorized_times = []
    opencl_times = []
    for i in range(len(Array_size)):
        mandelbrot_arr = mandelbrot_tools.generate_arrays(-2, 1, -1.5, 1.5, Array_size[i], np.longfloat, np.clongdouble)
        vectorized_times.append(vectorized(mandelbrot_arr, iterations))
        dask_vectorized_times.append(dask_vectorized(mandelbrot_arr, "384KB", iterations))
        opencl_times.append(mandelbrot_OpenCL(mandelbrot_arr, iterations))

    # create a 2D array from the values
    data = np.array([vectorized_times, dask_vectorized_times, opencl_times])

    # set up the plot
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=np.min(data), vmax=np.max(data))

    # add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # set the ticks and tick labels
    ax.set_xticks(np.arange(len(vectorized_times)))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(Array_size)
    ax.set_yticklabels(["vectorized", "dask_vectorized", "opencl"])

    # rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # add annotations
    for i in range(3):
        for j in range(len(vectorized_times)):
            text = ax.text(j, i, f"{data[i, j]:.5f}", ha="center", va="center", color="w")

    # set the title and show the plot
    ax.set_title("Execution Times (in seconds)")
    fig.tight_layout()
    plt.show()