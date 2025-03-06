import numpy
import matplotlib.pyplot as plt

def plot_mandelbrot(stable_members):
    plt.figure(figsize=(10, 10))
    plt.imshow(stable_members, cmap='hot', extent=(-2, 1, -1.5, 1.5))
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.show()

def stable_vectorized(c, threshold, num_iter):
    z = 0
    for _ in range(num_iter):
        z = z ** 2 + c
        if numpy.all(numpy.abs(z) >= 2): #shorten the execution time for the blocks located in regions where divergence is obtained quickly
            break
    return numpy.abs(z) <= threshold

#Lazy defines
def generate_arrays(rmin, rmax, imin, imax, pixel_density):
    real_part = numpy.linspace(rmin, rmax, pixel_density)
    imaginary_part = numpy.linspace(imin, imax, pixel_density)
    return real_part[numpy.newaxis, :] + imaginary_part[:, numpy.newaxis] * 1j

def generate_arrays_256(rmin, rmax, imin, imax, pixel_density):
    real_part = numpy.linspace(rmin, rmax, pixel_density, dtype=numpy.longdouble)
    imaginary_part = numpy.linspace(imin, imax, pixel_density, dtype=numpy.clongdouble)
    return real_part[numpy.newaxis, :] + imaginary_part[:, numpy.newaxis] * 1j

def generate_arrays_128(rmin, rmax, imin, imax, pixel_density):
    real_part = numpy.linspace(rmin, rmax, pixel_density, dtype=numpy.longdouble)
    imaginary_part = numpy.linspace(imin, imax, pixel_density, dtype=numpy.complex128)
    return real_part[numpy.newaxis, :] + imaginary_part[:, numpy.newaxis] * 1j

def generate_arrays_64(rmin, rmax, imin, imax, pixel_density):
    real_part = numpy.linspace(rmin, rmax, pixel_density, dtype=numpy.float64)
    imaginary_part = numpy.linspace(imin, imax, pixel_density, dtype=numpy.complex64)
    return real_part[numpy.newaxis, :] + imaginary_part[:, numpy.newaxis] * 1j

def generate_arrays_32(rmin, rmax, imin, imax, pixel_density):
    real_part = numpy.linspace(rmin, rmax, pixel_density, dtype=numpy.float32)
    imaginary_part = numpy.linspace(imin, imax, pixel_density, dtype=numpy.complex64)
    return real_part[numpy.newaxis, :] + imaginary_part[:, numpy.newaxis] * 1j

def generate_arrays_16(rmin, rmax, imin, imax, pixel_density):
    real_part = numpy.linspace(rmin, rmax, pixel_density, dtype=numpy.float16)
    imaginary_part = numpy.linspace(imin, imax, pixel_density, dtype=numpy.complex64)
    return real_part[numpy.newaxis, :] + imaginary_part[:, numpy.newaxis] * 1j