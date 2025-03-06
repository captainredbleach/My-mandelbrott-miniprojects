import multiprocessing
import time
from multiprocessing import Pool
import warnings
import mandelbrot_tools
import numpy
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def plot_mandelbrot(stable_members):
    plt.scatter(stable_members.real, stable_members.imag, color="black", marker=",", s=1)
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
def iterative(input_arr, iterations=100):
    mandelbrot = input_arr.copy()
    st = time.time()
    member = mandelbrot_tools.stable(mandelbrot, 2, iterations)
    end = time.time()
    elapsed_time = end - st
    print(f"{elapsed_time=} FOR 1 process iterative numpy")
    plot_mandelbrot(mandelbrot[member])
    


def vectorized(input_arr, iterations=100):
    mandelbrot = input_arr.copy()
    st = time.time()
    member = mandelbrot_tools.stable_vectorized(mandelbrot, 2, iterations)
    end = time.time()
    elapsed_time = end - st
    print(f"{elapsed_time=} FOR 1 process vectorized numpy")
    plot_mandelbrot(mandelbrot[member])


def vectorized_unrolled(input_arr, iterations=25):
    mandelbrot = input_arr.copy()
    st = time.time()
    member = mandelbrot_tools.stable_vectorized_unrolled(mandelbrot, 2, iterations//4)
    end = time.time()
    elapsed_time = end - st
    print(f"{elapsed_time=} FOR 1 process unrolled vectorized numpy")
    plot_mandelbrot(mandelbrot[member])

def iterative_numba(input_arr, iterations=100):
    mandelbrot = input_arr.copy()
    st = time.time()
    member = mandelbrot_tools.stable_numba(mandelbrot, 2, iterations)
    end = time.time()
    elapsed_time = end - st
    print(f"{elapsed_time=} FOR 1 process iterative numba")
    plot_mandelbrot(mandelbrot[member])


def vectorized_numba(input_arr, iterations=100):
    mandelbrot = input_arr.copy()
    st = time.time()
    member = mandelbrot_tools.stable_vectorized_numba(mandelbrot, 2, iterations)
    end = time.time()
    elapsed_time = end - st
    print(f"{elapsed_time=} FOR 1 process vectorized numba")
    plot_mandelbrot(mandelbrot[member])


def iterative_mp(input_arr, chunk_count=12, iterations=100):
    mandelbrot = input_arr.copy()
    split_chunks = numpy.split(mandelbrot, chunk_count)

    with Pool(multiprocessing.cpu_count()) as process_pool:
        arg_list = [(split_chunks[i], 2, iterations) for i in range(0, chunk_count)]
        st = time.time()
        ret = process_pool.starmap(mandelbrot_tools.stable, arg_list)
    end = time.time()
    elapsed_time = end - st
    print(f"Iterative {elapsed_time=} FOR {multiprocessing.cpu_count()} processes with {chunk_count=}")

    conc_ret = numpy.concatenate(ret)
    plot_mandelbrot(mandelbrot[conc_ret])


def vectorized_mp(input_arr, chunk_count=12, iterations=100):
    mandelbrot = input_arr.copy()
    split_chunks = numpy.split(mandelbrot, chunk_count)

    with Pool(multiprocessing.cpu_count()) as process_pool:
        arg_list = [(split_chunks[i], 2, iterations) for i in range(0, chunk_count)]
        st = time.time()
        ret = process_pool.starmap(mandelbrot_tools.stable_vectorized, arg_list)
    end = time.time()
    elapsed_time = end - st
    print(f"Vectorized {elapsed_time=} FOR {multiprocessing.cpu_count()} processes with {chunk_count=}")

    conc_ret = numpy.concatenate(ret)
    plot_mandelbrot(mandelbrot[conc_ret])

def iterative_mp_numba(input_arr, chunk_count=12, iterations=100):
    mandelbrot = input_arr.copy()
    split_chunks = numpy.split(mandelbrot, chunk_count)

    with Pool(multiprocessing.cpu_count()) as process_pool:
        arg_list = [(split_chunks[i], 2, iterations) for i in range(0, chunk_count)]
        st = time.time()
        ret = process_pool.starmap(mandelbrot_tools.stable_numba, arg_list)
    end = time.time()
    elapsed_time = end - st
    print(f"Iterative numba {elapsed_time=} FOR {multiprocessing.cpu_count()} processes with {chunk_count=}")

    conc_ret = numpy.concatenate(ret)
    plot_mandelbrot(mandelbrot[conc_ret])


def vectorized_mp_numba(input_arr, chunk_count=12, iterations=100):
    mandelbrot = input_arr.copy()
    split_chunks = numpy.split(mandelbrot, chunk_count)

    with Pool(multiprocessing.cpu_count()) as process_pool:
        arg_list = [(split_chunks[i], 2, iterations) for i in range(0, chunk_count)]
        st = time.time()
        ret = process_pool.starmap(mandelbrot_tools.stable_vectorized_numba, arg_list)
    end = time.time()
    elapsed_time = end - st
    print(f"Vectorized numba {elapsed_time=} FOR {multiprocessing.cpu_count()} processes with {chunk_count=}")
    conc_ret = numpy.concatenate(ret)
    plot_mandelbrot(mandelbrot[conc_ret])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mandelbrot_arr = mandelbrot_tools.generate_arrays(-2, 1, -1.5, 1.5, 1000)
    iterations = 100
    
    iterative(mandelbrot_arr, iterations)
    vectorized(mandelbrot_arr, iterations)
    vectorized_unrolled(mandelbrot_arr, iterations)
    iterative_numba(mandelbrot_arr, iterations)
    vectorized_numba(mandelbrot_arr, iterations)

    iterative_mp(mandelbrot_arr, multiprocessing.cpu_count(), iterations)
    iterative_mp(mandelbrot_arr, 2, iterations)
    iterative_mp(mandelbrot_arr, 4, iterations)
    iterative_mp(mandelbrot_arr, 10, iterations)
    iterative_mp(mandelbrot_arr, 100, iterations)
    iterative_mp(mandelbrot_arr, 1000, iterations)
    
    vectorized_mp(mandelbrot_arr, multiprocessing.cpu_count(), iterations)
    vectorized_mp(mandelbrot_arr, 2, iterations)
    vectorized_mp(mandelbrot_arr, 4, iterations)
    vectorized_mp(mandelbrot_arr, 10, iterations)
    vectorized_mp(mandelbrot_arr, 100, iterations)
    vectorized_mp(mandelbrot_arr, 1000, iterations)
    
    iterative_mp_numba(mandelbrot_arr, multiprocessing.cpu_count(), iterations)
    iterative_mp_numba(mandelbrot_arr, 2, iterations)
    iterative_mp_numba(mandelbrot_arr, 4, iterations)
    iterative_mp_numba(mandelbrot_arr, 10, iterations)
    iterative_mp_numba(mandelbrot_arr, 100, iterations)
    iterative_mp_numba(mandelbrot_arr, 1000, iterations)
    
    vectorized_mp_numba(mandelbrot_arr, multiprocessing.cpu_count(), iterations)
    vectorized_mp_numba(mandelbrot_arr, 2, iterations)
    vectorized_mp_numba(mandelbrot_arr, 4, iterations)
    vectorized_mp_numba(mandelbrot_arr, 10, iterations)
    vectorized_mp_numba(mandelbrot_arr, 100, iterations)
    vectorized_mp_numba(mandelbrot_arr, 1000, iterations)
