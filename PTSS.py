import numpy as np
from CPTS import CyPts as cptss
from CPTS import CyPtsJnt as cptsjnt
from CPTS import CyPtsSpt as cptsspt

# parsing for limb variables
class PtssJoint:
    def __init__(self, num_vert, num_horz):
        self.cpts = cptsjnt()
        self.num_vert = num_vert
        self.num_horz = num_horz

    # threshold at zero
    def thresh_fn(self, x):
        if x > 0:
            return x
        else:
            return 0

    # parse motor pattern into map code
    def parse(self, agn, ant, mus):
        num = self.num_vert
        size = num // 2
        map = self.cpts.parse_map(agn, ant, mus, num)
        return self.argmax(map, size)


    # sample map positions around map code
    def ptssjnt_gradient(self, b, a, b_max, a_max):
        if a == a_max + 1 and b == b_max:
            return 1 / 4
        if a == a_max - 1 and b == b_max:
            return 1 / 4
        if a == a_max and b == b_max + 1:
            return 1 / 4
        if a == a_max and b == b_max - 1:
            return 1 / 4
        if a == a_max and b == b_max:
            return 1
        else:
            return 0

    # bound map positions
    def ptssjnt_bound(self, b, a, b_bdry, a_bdry):
        return range(max(b - 1, 0), min(b + 1, b_bdry)), \
               range(max(a - 1, 0), min(a + 1, a_bdry))

    # check with zero array
    def test_zero(self, x):
        if np.array_equal(x, np.zeros(x.shape)):
            return 1
        else:
            return 0

    # extract index of maximal value
    def argmax(self, x, size):
        if self.test_zero(x) != 1:
            out = np.unravel_index(np.argmax(x), x.shape)
            return out
        else:
            return (size, size,)


# parsing general variables with 1 coordinate or 2 coordinates specified by type
class Ptss:
    def __init__(self, num_first, num_second):
        self.cpts = cptss()
        self.num_first = num_first
        self.num_second = num_second

    # threshold at zero
    def thresh_fn(self, x):
        if x > 0:
            return x
        else:
            return 0

    # compute agonist weights for map code
    def mus_agn(self, x, y, m):
        vector = np.array((x, y))
        if m == 0:
            weight = np.array((1, 0))
            agn1 = self.thresh_fn(np.dot(vector, weight))
            return agn1
        if m == 1:
            weight = np.array((0, 1))
            agn2 = self.thresh_fn(np.dot(vector, weight))
            return agn2


    # compute antagonist weights for map code
    def mus_ant(self, x, y, m):
        vector = np.array((x, y))
        if m == 0:
            weight = np.array((-1, 0))
            ant1 = self.thresh_fn(np.dot(vector, weight))
            return ant1
        if m == 1:
            weight = np.array((0, -1))
            ant2 = self.thresh_fn(np.dot(vector, weight))
            return ant2


    # compute weights for map code
    def mus_input(self, x, y):
        input = np.array([[self.mus_agn(x, y, m) for m in range(2)],
                          [self.mus_ant(x, y, m) for m in range(2)]])
        input = np.transpose(input, (1, 0))
        return input

    # convert map code into motor pattern
    def ret_mus_tran(self, x, y):
        num = self.num_first
        mus = self.mus_input(x, y) / num
        rev = self.rev_mus(mus)
        mus = 0.5 + mus - rev
        return mus

    # parse motor pattern into map code
    def parse(self, agn, ant, type):
        num = self.num_first
        size = num // 2
        map = self.cpts.parse_map(agn, ant, type, num)
        return self.argmax(map, size, type)

    # sample map positions around map code
    def ptss_gradient(self, a, b, a_max, b_max, type):
        if type == '1':
            if a == a_max + 1:
                return 1/2
            if a == a_max - 1:
                return 1/2
            if a == a_max:
                return 1
            else:
                return 0

        if type == '2':
            if a == a_max + 1 and b == b_max:
                return 1 / 4
            if a == a_max - 1 and b == b_max:
                return 1 / 4
            if a == a_max and b == b_max + 1:
                return 1 / 4
            if a == a_max and b == b_max - 1:
                return 1 / 4
            if a == a_max and b == b_max:
                return 1
            else:
                return 0

    # bound map positions
    def ptss_bound(self, a, b, a_bdry, b_bdry, type):
        if type == '1':
            return range(max(a - 1, 0), min(a + 1, a_bdry))

        if type == '2':
            return range(max(a - 1, 0), min(a + 1, a_bdry)), \
                   range(max(b - 1, 0), min(b + 1, b_bdry))

    # check with zero array
    def test_zero(self, x):
        if np.array_equal(x, np.zeros(x.shape)):
            return 1
        else:
            return 0

    # extract index of maximal activity
    def argmax(self, x, size, type):
        if type == '1':
            if self.test_zero(x) != 1:
                out = np.unravel_index(np.argmax(x), x.shape)
                return out
            else:
                return (size,)

        if type == '2':
            if self.test_zero(x) != 1:
                out = np.unravel_index(np.argmax(x), x.shape)
                return out
            else:
                return (size, size,)

    # switch agonist and antagonist
    def rev_mus(self, x):
        mus = np.zeros((2, 2))
        for s in range(2):
            for n in range(2):
                mus[s][n] = x[s][(n + 1) % 2]
        return mus


# parsing general variables with 3 coordinates
class PtssSpatial:
    def __init__(self, num_horz, num_vert, num_rads):
        self.cpts = cptsspt
        self.num_horz = num_horz
        self.num_vert = num_vert
        self.num_rads = num_rads

    # threshold at zero
    def thresh_fn(self, x):
        if x > 0:
            return x
        else:
            return 0

    # parse motor pattern into map code
    def parse(self, agn, ant):
        num = self.num_vert
        size = num // 2
        map = self.cpts.parse_map(agn, ant, num)
        return self.argmax(map, size)


    # sample map positions around map code
    def ptssspt_gradient(self, a, b, r, a_max, b_max, r_max):
        if a == a_max + 1 and b == b_max and r == r_max:
            return 1 / 6
        if a == a_max - 1 and b == b_max and r == r_max:
            return 1 / 6
        if a == a_max and b == b_max + 1 and r == r_max:
            return 1 / 6
        if a == a_max and b == b_max - 1 and r == r_max:
            return 1 / 6
        if a == a_max and b == b_max and r == r_max + 1:
            return 1 / 6
        if a == a_max and b == b_max and r == r_max - 1:
            return 1 / 6
        if a == a_max and b == b_max and r == r_max:
            return 1
        else:
            return 0

    # bound map positions
    def ptssspt_bound(self, a, b, r, a_bdry, b_bdry, r_bdry):
        return np.arange(max(a - 1, 0), min(a + 1, a_bdry), dtype=int), \
               np.arange(max(b - 1, 0), min(b + 1, b_bdry), dtype=int), \
               np.arange(max(r - 1, 0), min(r + 1, r_bdry), dtype=int)

    # check with zero array
    def test_zero(self, x):
        if np.array_equal(x, np.zeros(x.shape)):
            return 1
        else:
            return 0

    # extract index of maximal activity
    def argmax(self, x, size):
        if self.test_zero(x) != 1:
            out = np.unravel_index(np.argmax(x), x.shape)  # format is (height, width)
            return out
        else:
            return (size, size, size,)



