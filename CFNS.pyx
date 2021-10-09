import numpy as np
cimport cython
from libc.math cimport sqrt, exp

# C extension class for FUNCS Module
cdef class CyFns:

    # filter binocular map with nearest neighbor kernel
    def binoc_map(self, double[:, :, :] map, int size):
        out = np.zeros((2 * size, 2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :, :] out_view = out
        cdef double[:, :, :] map_view = map
        cdef Py_ssize_t d, j, i, q, p
        for d in range(2 * size):
            for j in range(2 * size):
                for i in range(2 * size):
                    for q in range(max(j - 1, 0), min(j + 1, 2 * size)):
                        for p in range(max(i - 1, 0), min(i + 1, 2 * size)):
                            out_view[d][j][i] += map_view[d][q][p] * 1/4
        return out

    # bound index array
    cdef int retmap_bound(self, int x, int size):
        if x < 0:
            return 0
        if x > size - 1:
            return size - 1
        else:
            return x

    # form binocular map from fusing left and right retinal maps
    def binoc_fuse(self, double[:, :] y, double[:, :] x, int size):
        out = np.zeros((2 * size, 2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :, :] out_view = out
        cdef double[:, :] x_view = x
        cdef double[:, :] y_view = y
        cdef Py_ssize_t d, j, i, i_L, i_R
        for d in range(-size, size):
            for j in range(-size, size):
                for i in range(-size, size):
                    i_L = self.retmap_bound(i + d + size, 2 * size)
                    i_R = self.retmap_bound(i - d + size, 2 * size)
                    out_view[d + size][j + size][i + size] = (y_view[j][i_L] + x_view[j][i_R]) / 2
        return out

    # form retinal map centered at foveae
    def gaze_map(self, int[:] pt, double[:, :] map, int size):
        gaze = np.zeros((4 * size, 4 * size), dtype=np.float64)
        cdef double[:, :] gaze_view = gaze
        cdef int[:] pt_view = pt
        cdef double[:, :] map_view = map
        cdef Py_ssize_t j, i
        for j in range(2 * size):
            for i in range(2 * size):
                if (j, i) != tuple(pt_view):
                    gaze_view[(j - pt_view[0]) + 2 * size][(i - pt_view[1]) + 2 * size] = map_view[j][i]

        out = np.zeros((2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :] out_view = out
        for j in range(2 * size):
            for i in range(2 * size):
                out_view[j][i] = gaze_view[size + j][size + i]
        return out

    # 2d gaussian kernel
    cdef double gauss_gradient(self, int q, int p, int j, int i, double sigma):
        cdef double r, grad
        r = sqrt((q - j) ** 2 + (p - i) ** 2)
        grad = exp(-sigma * r ** 2)
        return grad

    # 2d gaussian norm
    cdef double gauss_norm(self, int j, int i, double sigma, int size):
        cdef double norm = 0
        cdef Py_ssize_t q, p
        for q in range(max(j - 1, 0), min(j + 1, size)):
            for p in range(max(i - 1, 0), min(i + 1, size)):
                norm += self.gauss_gradient(q, p, j, i, sigma)
        return norm

    # check at some value
    cdef int delta_fn(self, int x, int a):
        if (x == a) == True:
            return 1
        else:
            return 0

    # check at some index
    cdef int index_fn(self, int j, int i, int b, int a):
        return 1 - self.delta_fn(j, b) * self.delta_fn(i, a)

    # filter retinal map with gaussian kernel in the center
    @cython.cdivision(True)
    def sc_map_cent(self, double[:, :] map, double sigma, int size):
        out = np.zeros((2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t j, i, q, p
        cdef double accum
        for j in range(2 * size):
            for i in range(2 * size):
                accum = 0
                for q in range(max(j - 1, 0), min(j + 1, 2 * size)):
                    for p in range(max(i - 1, 0), min(i + 1, 2 * size)):
                        accum += map_view[q][p] * self.gauss_gradient(q, p, j, i, sigma)
                out_view[j][i] = accum * self.index_fn(j, i, size, size) / self.gauss_norm(j, i, sigma, 2 * size)
        return out

    # filter retinal map with gaussian kernel in the surround
    @cython.cdivision(True)
    def sc_map_surr(self, double[:, :] map, double sigma, int size):
        out = np.zeros((2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t j, i, q, p
        cdef double accum
        for j in range(2 * size):
            for i in range(2 * size):
                accum = 0
                for q in range(max(j - 1, 0), min(j + 1, 2 * size)):
                    for p in range(max(i - 1, 0), min(i + 1, 2 * size)):
                        accum += map_view[q][p] * self.gauss_gradient(q, p, j, i, sigma) * self.index_fn(q, p, j, i)
                out_view[j][i] = accum * self.index_fn(j, i, size, size) / self.gauss_norm(j, i, sigma, 2 * size)
        return out

