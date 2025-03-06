import numpy as np
import pyopencl as cl
import unittest
from unittest.mock import patch
from io import StringIO
import matplotlib.pyplot as plt
from mandelbrot_tools import opencl_vectorized
from mandelbrot_tools import stable_vectorized
from mandelbrot_tools import generate_arrays
from mandelbrot_tools import plot_mandelbrot

class TestOpenCLVectorized(unittest.TestCase):
    def test_opencl_vectorized(self):
        # test inputs
        input_arr = np.array([[-2, -1], [-1, 0], [0, 1]], dtype=np.complex64)
        num_iter = 50
        ctx = cl.create_some_context()
        
        # expected output
        expected_output = np.array(
            [[1, 1], [1, 1], [0, 1]], dtype=np.uint16).reshape(input_arr.shape)
        
        # test output and elapsed time
        output, elapsed_time = opencl_vectorized(input_arr, num_iter, ctx)
        
        # check output values
        np.testing.assert_array_equal(output, expected_output)
        
        # check output data type
        self.assertEqual(output.dtype, np.uint16)
        
        # check elapsed time
        self.assertIsInstance(elapsed_time, float)
        
class TestStableVectorized(unittest.TestCase):
    
    def test_output_shape(self):
        c = np.zeros((100, 100))
        threshold = 2
        num_iter = 100
        output = stable_vectorized(c, threshold, num_iter)
        self.assertEqual(output.shape, c.shape)
    
    def test_input_types(self):
        c = np.zeros((100, 100))
        threshold = 2
        num_iter = 100
        self.assertRaises(TypeError, stable_vectorized, "not an ndarray", threshold, num_iter)
        self.assertRaises(TypeError, stable_vectorized, c, "not an int", num_iter)
        self.assertRaises(TypeError, stable_vectorized, c, threshold, "not an int")
    
    def test_invalid_input_values(self):
        with self.assertRaises(ValueError):
            stable_vectorized(np.array([1, 2, 3]), 2, 10)
        with self.assertRaises(ValueError):
            stable_vectorized(np.array([[1, 2], [3, 4], [5, 6]]), 2, 10)
        with self.assertRaises(ValueError):
            stable_vectorized(np.array([[1+2j, 2-3j], [3+4j, 4-5j]]), 2, 10)
        with self.assertRaises(ValueError):
            stable_vectorized(np.array([[-2.0, 1.0], [0.5, 0.0], [1.5, -1.0]]), -1, 10)
        
class TestGenerateArrays(unittest.TestCase):
    
    def test_output_shape(self):
        rmin, rmax, imin, imax, pixel_density = -2, 2, -2, 2, 100
        output = generate_arrays(rmin, rmax, imin, imax, pixel_density)
        expected_shape = (pixel_density, pixel_density)
        self.assertEqual(output.shape, expected_shape)
    
    def test_output_types(self):
        rmin, rmax, imin, imax, pixel_density = -2, 2, -2, 2, 100
        output = generate_arrays(rmin, rmax, imin, imax, pixel_density, real_type=np.float32, complex_type=np.complex64)
        self.assertTrue(output.dtype == np.complex64)
        self.assertTrue(output.real.dtype == np.float32)
        self.assertTrue(output.imag.dtype == np.float32)
    
    def test_invalid_input_values(self):
        rmin, rmax, imin, imax, pixel_density = 2, -2, 2, -2, 100
        self.assertRaises(ValueError, generate_arrays, rmin, rmax, imin, imax, pixel_density)
        
class TestPlotMandelbrot(unittest.TestCase):
    def setUp(self):
        self.input_arr = generate_arrays(-2, 1, -1.5, 1.5, 100)
        self.stable_members, _ = stable_vectorized(self.input_arr, 2, 100)

    @patch('sys.stdout', new_callable=StringIO)
    def test_plot_mandelbrot(self, fake_output):
        expected_output_type = type(plt.Figure())
        plot_mandelbrot(self.stable_members)
        self.assertEqual(type(fake_output), StringIO)
        self.assertIsInstance(plt.gcf(), expected_output_type)

if __name__ == '__main__':
    unittest.main()