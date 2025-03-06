import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
import time

def plot_mandelbrot(stable_members):
    """
    Plots the mandelbrot array

    Parameters
    ----------
        stable_members (ndarray): The binary input array
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(stable_members, cmap='hot', extent=(-2, 1, -1.5, 1.5))
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.show()

def stable_vectorized(c, threshold, num_iter):
    """
    Approximate the points using numpy vectorized method within the passed array and include it if they belong to the fractal.

    Parameters
    ----------
        c (ndarray): The input array
        threshold (int): The threshold to determine the approximated point
        num_iter (int): Number of iterations done for calculation

    Returns
    -------
        ndarray: Binary array containing information if a specific point belongs to the fractal
    """
    if not isinstance(c, np.ndarray.astype):
        raise ValueError("c must be a numpy array")
    if not isinstance(threshold, int) or threshold < 0:
        raise ValueError("threshold must be a non-negative integer")
    if not isinstance(num_iter, int) or num_iter < 0:
        raise ValueError("num_iter must be a non-negative integer")
    z = 0
    for _ in range(num_iter):
        z = z ** 2 + c
        if np.all(np.abs(z) >= 2): #shorten the execution time for the blocks located in regions where divergence is obtained quickly
            break
    return np.abs(z) <= threshold


def opencl_vectorized(input_arr, num_iter, ctx):
    """
    Approximate the points using opencl vectorized method within the passed array and include it if they belong to the fractal.

    Parameters
    ----------
        input_arr (ndarray): The input array
        num_iter (int): Number of iterations done for calculation
        ctx: The device context
        
    Returns
    -------
        ndarray: Binary array containing information if a specific point belongs to the fractal
    """
    if not isinstance(input_arr, np.ndarray):
        raise ValueError("c must be a numpy array")
    if not isinstance(num_iter, int) or num_iter < 0:
        raise ValueError("num_iter must be a non-negative integer")
    queue = cl.CommandQueue(ctx)
    q = np.ravel(input_arr).astype(np.complex64)
    output = np.empty(q.shape, dtype=np.uint16)

    mf = cl.mem_flags
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

    prg = cl.Program(
        ctx,
        """
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void mandelbrot(__global float2 *q,
                     __global ushort *output, ushort const num_iter)
    {
        int gid = get_global_id(0);
        float z, real = 0;
        float imag = 0;
        output[gid] = 0;
        for(int i = 0; i < num_iter; i++) {
            z = real*real - imag*imag + q[gid].x;
            imag = 2* real*imag + q[gid].y;
            real = z;
            if (real*real + imag*imag > 2.0f) {
                 output[gid] = i;
                 break;
            }
        }
    }
    """,
    ).build()
    st = time.time()
    prg.mandelbrot(
        queue, output.shape, None, q_opencl, output_opencl, np.uint16(num_iter)
    )
    end = time.time()
    cl.enqueue_copy(queue, output, output_opencl).wait()
    elapsed_time = end - st
    output[np.absolute(output) >= 1] = 1
    return output.reshape(input_arr.shape), elapsed_time


def generate_arrays(rmin, rmax, imin, imax, pixel_density, real_type=None, complex_type=None):
    """
    Generate Numpy arrays with specified size, pixel density in datatypes.

    Parameters
    ----------
    rmin (float):Real part lower bound.
    rmax (float):Real part upper bound.
    imin (float):Imaginary part lower bound.
    imax (float): Imaginary part upper bound.
    pixel_density: The size of the array
    real_type: Datatype used for real part calculations.
    complex_type: Datatype used for complex part calculations.

    Returns
    -------
    numpy.array
        Generated numpy array ready to be used for the calculations.
    """
    if rmin >= rmax:
        raise ValueError("rmin must be less than rmax")
    if imin >= imax:
        raise ValueError("imin must be less than imax")
    real_part = np.linspace(rmin, rmax, pixel_density, dtype=real_type)
    imaginary_part = np.linspace(imin, imax, pixel_density,dtype=complex_type)
    return (real_part[np.newaxis, :] + imaginary_part[:, np.newaxis] * 1j).astype(complex_type)


        
