import numpy
import numba

def stable(c, threshold, num_iter):
    stability_arr = numpy.zeros(c.shape, dtype=bool)
    for r in range(c.shape[0]):
        for i in range(c.shape[1]):
            checked_point = c[r, i]
            z = 0
            for _ in range(num_iter):
                z = z ** 2 + checked_point
            stability_arr[r, i] = True if abs(z) <= threshold else False
    return stability_arr

def stable_vectorized(c, threshold, num_iter):
    z = 0
    for _ in range(num_iter):
        z = z ** 2 + c
    return abs(z) <= threshold

def stable_vectorized_unrolled(c, threshold, num_iter):
    z = 0
    for _ in range(num_iter):
        z = z ** 2 + c
        z = z ** 2 + c
        z = z ** 2 + c
        z = z ** 2 + c
    return abs(z) <= threshold

@numba.jit()
def stable_numba(c, threshold, num_iter):
    stability_arr = numpy.zeros(c.shape, dtype=bool)
    for r in range(c.shape[0]):
        for i in range(c.shape[1]):
            checked_point = c[r, i]
            z = 0
            for _ in range(num_iter):
                z = z ** 2 + checked_point
            stability_arr[r, i] = True if abs(z) <= threshold else False
    return stability_arr


@numba.jit()
def stable_vectorized_numba(c, threshold, num_iter):
    z = 0
    for _ in range(num_iter):
        z = z ** 2 + c
    return abs(z) <= threshold

def generate_arrays(rmin, rmax, imin, imax, pixel_density):
    real_part = numpy.linspace(rmin, rmax, int((rmax - rmin) * pixel_density))
    imaginary_part = numpy.linspace(imin, imax, int((imax - imin) * pixel_density))
    return real_part[numpy.newaxis, :] + imaginary_part[:, numpy.newaxis] * 1j
