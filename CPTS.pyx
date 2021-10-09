import numpy as np
cimport cython
from libc.math cimport sqrt


# C extension class for PTSS Module

# parsing for limb variables
cdef class CyPtsJnt:

    # form dot product
    cdef double dot(self, double[:] x, double[:] y):
        cdef double out = 0
        cdef double[:] x_view = x
        cdef double[:] y_view = y
        cdef int dim = x.shape[0]
        cdef Py_ssize_t i
        for i in range(dim):
            out += x_view[i] * y_view[i]
        return out

    # threshold at zero
    cdef double thresh(self, double x):
        if x >= 0:
            return x
        else:
            return 0

    # rescale variable
    @cython.cdivision(True)
    cdef double rescale(self, int x, int num):
        cdef double value
        value = 2 * (float(x) / float(num)) - 1
        return value

    # multiply agonist pattern by weights
    cdef double ptss_jnt_agn(self, double z, double b, double a, int m):
        cdef double[:] vector = np.array((b, a), dtype=np.float64)
        cdef double[:] weight1 = np.array((1, 0), dtype=np.float64)
        cdef double[:] weight2 = np.array((0, 1), dtype=np.float64)
        cdef double agn1, agn2

        if m == 0:
            agn1 = self.thresh(z) * self.thresh(self.dot(vector, weight1))
            return agn1
        if m == 1:
            agn2 = self.thresh(z) * self.thresh(self.dot(vector, weight2))
            return agn2

    # multiply antagonist pattern by weights
    cdef double ptss_jnt_ant(self, double z, double b, double a, int m):
        cdef double[:] vector = np.array((b, a), dtype=np.float64)
        cdef double[:] weight1 = np.array((-1, 0), dtype=np.float64)
        cdef double[:] weight2 = np.array((0, -1), dtype=np.float64)
        cdef double ant1, ant2

        if m == 0:
            ant1 = self.thresh(z) * self.thresh(self.dot(vector, weight1))
            return ant1
        if m == 1:
            ant2 = self.thresh(z) * self.thresh(self.dot(vector, weight2))
            return ant2

    # parse motor pattern into map activity
    def parse_map(self, double[:] agn, double[:] ant, int mus, int num):
        out = np.zeros((num, num), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:] agn_view = agn
        cdef double[:] ant_view = ant
        cdef Py_ssize_t b, a, m
        cdef double new_b, new_a, vector, accum
        for b in range(num):
            for a in range(num):
                accum = 0
                new_b = self.rescale(b, num)
                new_a = self.rescale(a, num)
                vector = sqrt(new_b**2 + new_a**2)
                for m in range(mus):
                    accum += self.ptss_jnt_agn(agn_view[m] - ant_view[m], new_b, new_a, m) + \
                    self.ptss_jnt_ant(ant_view[m] - agn_view[m], new_b, new_a, m)
                out_view[b][a] = self.thresh(accum - 0.5 * vector ** 2)
        return out

# parsing for general variables with 1 or 2 coordinates depending on type
cdef class CyPts:

    # form dot product
    cdef double dot(self, double[:] x, double[:] y):
        cdef double out = 0
        cdef double[:] x_view = x
        cdef double[:] y_view = y
        cdef int dim = x.shape[0]
        cdef Py_ssize_t i
        for i in range(dim):
            out += x_view[i] * y_view[i]
        return out

    # threshold at zero
    cdef double thresh(self, double x):
        if x >= 0:
            return x
        else:
            return 0

    # rescale variable
    @cython.cdivision(True)
    cdef double rescale(self, int x, int num):
        cdef double value
        value = 2 * (float(x) / float(num)) - 1
        return value

    # multiply agonist pattern with weights
    cdef double ptss_pt_agn(self, double z, double a, double b, int k, str type):
        cdef double agn

        if type == '1':
            agn = self.thresh(z) * self.thresh(a * 1)
            return agn

        cdef double[:] vector = np.array((a, b), dtype=np.float64)
        cdef double[:] weight1 = np.array((1, 0), dtype=np.float64)
        cdef double[:] weight2 = np.array((0, 1), dtype=np.float64)
        cdef double agn1, agn2

        if type == '2':
            if k == 0:
                agn1 = self.thresh(z) * self.thresh(self.dot(vector, weight1))
                return agn1
            if k == 1:
                agn2 = self.thresh(z) * self.thresh(self.dot(vector, weight2))
                return agn2

    # multiply antagonist pattern with weights
    cdef double ptss_pt_ant(self, double z, double a, double b, int k, str type):
        cdef double ant

        if type == '1':
            ant = self.thresh(z) * self.thresh(a * -1)
            return ant

        cdef double[:] vector = np.array((a, b), dtype=np.float64)
        cdef double[:] weight1 = np.array((-1, 0), dtype=np.float64)
        cdef double[:] weight2 = np.array((0, -1), dtype=np.float64)
        cdef double ant1, ant2

        if type == '2':
            if k == 0:
                ant1 = self.thresh(z) * self.thresh(self.dot(vector, weight1))
                return ant1
            if k == 1:
                ant2 = self.thresh(z) * self.thresh(self.dot(vector, weight2))
                return ant2

    # parse motor pattern into map activity
    def parse_map(self, double[:] agn, double[:] ant, str type, int num):
        out1 = np.zeros((num), dtype=np.float64)
        cdef double[:] out1_view = out1
        cdef Py_ssize_t a1
        cdef double new_a1, vector1, accum1

        if type == '1':
            for a1 in range(num):
                accum1 = 0
                new_a1 = self.rescale(a1, num)
                vector1 = sqrt(new_a1**2)
                accum1 = self.ptss_pt_agn(agn[0] - ant[0], new_a1, 0, 0, '1') + \
                self.ptss_pt_ant(ant[0] - agn[0], new_a1, 0, 0, '1')
                out1_view[a1] = self.thresh(accum1 - 0.5 * vector1 ** 2)

            return out1

        out2 = np.zeros((num, num), dtype=np.float64)
        cdef double[:, :] out2_view = out2
        cdef double[:] agn_view = agn
        cdef double[:] ant_view = ant
        cdef Py_ssize_t a2, b2, k
        cdef double new_a2, new_b2, vector2, accum2

        if type == '2':
            for a2 in range(num):
                for b2 in range(num):
                    accum2 = 0
                    new_a2 = self.rescale(a2, num)
                    new_b2 = self.rescale(b2, num)
                    vector2 = sqrt(new_a2**2 + new_b2**2)
                    for k in range(2):
                        accum2 += self.ptss_pt_agn(agn_view[k] - ant_view[k], new_a2, new_b2, k, '2') + \
                        self.ptss_pt_ant(ant_view[k] - agn_view[k], new_a2, new_b2, k, '2')
                    out2_view[a2][b2] = self.thresh(accum2 - 0.5 * vector2 ** 2)
            return out2


# parsing for spatial variables
cdef class CyPtsSpt:

    # form dot product
    cdef double dot(self, double[:] x, double[:] y):
        cdef double out = 0
        cdef double[:] x_view = x
        cdef double[:] y_view = y
        cdef int dim = x.shape[0]
        cdef Py_ssize_t i
        for i in range(dim):
            out += x_view[i] * y_view[i]
        return out

    # threshold at zero
    cdef double thresh(self, double x):
        if x >= 0:
            return x
        else:
            return 0

    # rescale variable
    @cython.cdivision(True)
    cdef double rescale(self, int x, int num):
        cdef double value
        value = 2 * (float(x) / float(num)) - 1
        return value

    # multiply agonist pattern with weights
    cdef double ptss_pt_agn(self, double z, double a, double b, double r, int k):
        cdef double[:] vector = np.array((a, b, r), dtype=np.float64)
        cdef double[:] weight1 = np.array((1, 0, 0), dtype=np.float64)
        cdef double[:] weight2 = np.array((0, 1, 0), dtype=np.float64)
        cdef double[:] weight3 = np.array((0, 0, 1), dtype=np.float64)
        cdef double agn1, agn2, agn3

        if k == 0:
            agn1 = self.thresh(z) * self.thresh(self.dot(vector, weight1))
            return agn1

        if k == 1:
            agn2 = self.thresh(z) * self.thresh(self.dot(vector, weight2))
            return agn2

        if k == 2:
            agn3 = self.thresh(z) * self.thresh(self.dot(vector, weight3))
            return agn3

    # multiply antagonist pattern with weights
    cdef double ptss_pt_ant(self, double z, double a, double b, double r, int k):
        cdef double[:] vector = np.array((a, b, r), dtype=np.float64)
        cdef double[:] weight1 = np.array((-1, 0, 0), dtype=np.float64)
        cdef double[:] weight2 = np.array((0, -1, 0), dtype=np.float64)
        cdef double[:] weight3 = np.array((0, 0, -1), dtype=np.float64)
        cdef double ant1, ant2, ant3

        if k == 0:
            ant1 = self.thresh(z) * self.thresh(self.dot(vector, weight1))
            return ant1

        if k == 1:
            ant2 = self.thresh(z) * self.thresh(self.dot(vector, weight2))
            return ant2

        if k == 2:
            ant3 = self.thresh(z) * self.thresh(self.dot(vector, weight3))
            return ant3

    # parse motor pattern into map activity
    def parse_map(self, double[:] agn, double[:] ant, int num):
        out = np.zeros((num, num, num), dtype=np.float64)
        cdef double[:, :, :] out_view = out
        cdef double[:] agn_view = agn
        cdef double[:] ant_view = ant
        cdef Py_ssize_t a, b, r, k
        cdef double new_a, new_b, new_r, vector, accum

        for a in range(num):
            for b in range(num):
                for r in range(num):
                    accum = 0
                    new_a = self.rescale(a, num)
                    new_b = self.rescale(b, num)
                    new_r = self.rescale(r, num)
                    vector = sqrt(new_a**2 + new_b**2 + new_r**2)
                    for k in range(3):
                        accum += self.ptss_pt_agn(agn_view[k] - ant_view[k], new_a, new_b, new_r, k) + \
                        self.ptss_pt_ant(ant_view[k] - agn_view[k], new_a, new_b, new_r, k)
                    out_view[a][b][r] = self.thresh(accum - 0.5 * vector ** 2)
        return out



cdef class CyPtsEye:
    pass




