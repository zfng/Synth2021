import numpy as np
from PTSS import PtssJoint as ptssjnt
from PTSS import Ptss as ptss
from CFNS import CyFns as cfns


# define the 4th order Runge-Kutta algorithm.
class RK4:
    def rk4(self, y0, dy, step):
        k1 = step * dy
        k2 = step * (dy + 1 / 2 * k1)
        k3 = step * (dy + 1 / 2 * k2)
        k4 = step * (dy + k3)
        y1 = y0 + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return y1


# list the functions used in the modules
class FNS:
    def __init__(self):
        self.cfns = cfns()
    # ____________________________________________________________________________________________________________
    # Common Functions

    # threshold at some value
    def thresh_fn(self, x, thresh):
        return np.sign(x - thresh) * (x - thresh) * self.indic_fn(x - thresh)

    # bound within some interval
    def bound_fn(self, x, thresh):
        rightbd = np.heaviside(thresh - x, 0)
        out = x * rightbd + thresh * (rightbd + 1) % 2
        leftbd = np.heaviside(out - -thresh, 0)
        out = out * leftbd + -thresh * ((leftbd + 1) % 2)
        return out

    # cutoff outside some interval
    def cutoff_fn(self, x, thresh):
        rightbd = np.heaviside(x - thresh, 0)
        rightout = x * rightbd
        leftbd = np.heaviside(-x - thresh, 0)
        leftout = x * leftbd
        return rightout + leftout

    # check at some value
    def delta_fn(self, x, a):
        if np.all(x == a) == True:
            return 1
        else:
            return 0

    # check within some interval
    def cond_fn(self, x, a):
        out = np.heaviside(a - x, 0) * np.heaviside(x - -a, 0)
        return out

    # check at some index
    def index_fn(self, j, i, b, a):
        return 1 - self.delta_fn(j, b) * self.delta_fn(i, a)

    # check sign at zero
    def indic_fn(self, x):
        return np.heaviside(x, 0)

    # binary sampling function
    def sample_fn(self, x, thresh):
        return 1 * self.indic_fn(x - thresh)

    # sigmoid sampling function
    def sigmoid_fn(self, x, offset, power):
        return x**power / (offset**power + x**power)

    # enlarge array size
    def enlarge(self, x, y, stride, num, type):
        # given x is large and y is small and type is number of relevant dimensions
        if type == '3':
            for a in range(0, num, stride):
                for b in range(0, num, stride):
                    for c in range(0, num, stride):
                        new_a = a // stride
                        new_b = b // stride
                        new_c = c // stride
                        x[a][b][c] = y[new_a][new_b][new_c]
            return x

        if type == '2':
            for a in range(0, num, stride):
                for b in range(0, num, stride):
                    new_a = a // stride
                    new_b = b // stride
                    x[a][b] = y[new_a][new_b]
            return x

    # shrink array size
    def shrink(self, x, y, stride, num, type):
        # given x is large and y is small and type is number of relevant dimensions
        new_num = num // stride
        if type == '3':
            for a in range(0, new_num):
                for b in range(0, new_num):
                    for c in range(0, new_num):
                        new_a = self.index_bound(a * stride, num)
                        new_b = self.index_bound(b * stride, num)
                        new_c = self.index_bound(c * stride, num)
                        y[a][b][c] = x[new_a][new_b][new_c]
            return y

        if type == '2':
            for a in range(0, new_num):
                for b in range(0, new_num):
                    new_a = self.index_bound(a * stride, num)
                    new_b = self.index_bound(b * stride, num)
                    y[a][b] = x[new_a][new_b]
            return y

    # bound index
    def index_bound(self, x, size):
        if x < size:
            return x
        else:
            return size - 1


    # ____________________________________________________________________________________________________________
    # EYES Module

    # bound array index
    def retmap_bound(self, x, size):
        if x < 0:
            return 0
        if x > size - 1:
            return size - 1
        else:
            return x

    # check if maximal value of array is not at center
    def fixate(self, gaz_map, size):
        fix_map = np.ones((2, 2 * size, 2 * size))
        for s in range(2):
            fix_map[s][size, size] = 1
        if np.array_equal(fix_map, gaz_map) == True:
            return 0
        else:
            return 1

    # compute difference btw agonist and antagonist for learning variables
    def diff_mat(self, x, size):
        mat = np.zeros((2, 2, 2, size, size))
        for s in range(2):
            for m in range(2):
                mat[s][m] = x[s][m][0] - x[s][m][1], x[s][m][1] - x[s][m][0]
        return mat

    # check epoch within an interval in the forward direction
    def forwd_period(self, t, T, interval):
        if (t // interval) * interval + 0 <= t and t < (t // interval) * interval + T:
            return 1
        else:
            return 0

    # check epoch within an interval in the backward direction
    def backw_period(self, t, T, interval):
        if (t // interval + 1) * interval - T <= t and t < (t // interval + 1) * interval + 0:
            return 1
        else:
            return 0

    # list epochs within some interval
    def intv_period(self, t, interval):
        lb = (t // interval) * interval
        ub = (t // interval + 1) * interval
        return np.arange(lb, ub, 1)

    # list epoch-value pairs within some interval
    def add_error(self, z, t, interval):
        range = self.intv_period(t, interval)
        value = [z] * interval
        add = [(x, y) for x, y in zip(range, value)]
        return add

    # check if equal to the zero array
    def test_zero(self, x):
        if np.array_equal(x, np.zeros(x.shape)):
            return 1
        else:
            return 0

    # extract index of maximal value for an array centered at zero
    def argmax(self, x, size):
        if self.test_zero(x) != 1:
            out = np.array(np.unravel_index(np.argmax(x), x.shape)) - size  # format is (height, width)
            return out
        else:
            return np.zeros(2)

    # populate in a neighborhood around the given index
    def arrmax(self, max, size):
        ptts = ptss(2 * size, 2 * size)
        out = np.zeros((2, 2 * size, 2 * size))

        for s in range(2):
            b_max, a_max = np.array(max[s], dtype=int)
            bound = ptts.ptss_bound(b_max, a_max, 2 * size, 2 * size, '2')
            for b in bound[0]:
                for a in bound[1]:
                    out[s][b][a] = ptts.ptss_gradient(b, a, b_max, a_max, '2')
        return out

    # compute sum btw agonist and antagonist
    def sum_mus(self, x):
        mus = np.zeros((2))
        for s in range(2):
            mus[s] = np.sum(x[s])
        return mus

    # extract agonist
    def extract_ang(self, x):
        out = np.zeros(3)
        for k in range(3):
            out[k] = x[k][0]
        return out

    # convert normalized activity into angle for eye variables
    def conv_targ(self, x):
        # for eye movement and representation
        ang_rang = 1.0 * np.radians([-45, 45])
        dist_rang = 1.0 * np.array((5, 50))

        # -----------------------------------------------------------------------------------------------------------
        # for eye-leg/eye-hand coordination
        #ang_rang = 2.0 * np.radians([-45, 45])
        #dist_rang = 2.5 * np.array((5, 50))

        # -----------------------------------------------------------------------------------------------------------
        est = self.extract_ang(x)

        horz = ang_rang[0] + (ang_rang[1] - ang_rang[0]) * est[0]
        vert = ang_rang[0] + (ang_rang[1] - ang_rang[0]) * est[1]

        a = dist_rang[0] / (dist_rang[1] - dist_rang[0])
        b = (1 + a) * dist_rang[0]
        dist = b / (a + est[2])

        return np.array((np.degrees(horz), np.degrees(vert), dist))

    # parse eye variables
    def parse_eye(self, mus, size):
        ptts = ptss(2 * size, 2 * size)
        left, right = mus

        left_agn, left_ant = np.transpose(left, (1, 0))
        right_agn, right_ant = np.transpose(right, (1, 0))

        left_vert, left_horz = ptts.parse(left_agn, left_ant, '2')
        right_vert, right_horz = ptts.parse(right_agn, right_ant, '2')

        return np.array([(left_vert, left_horz), (right_vert, right_horz)], dtype=int)

    # parse general variables with 3 coordinates
    def parse_targ(self, mus, num_deg, type):
        if type == 'eye':
            ang_parse = self.parse_coord(mus[0:2], num_deg, '2')
            dist_parse = self.parse_coord(mus[2], num_deg, '1')
            parse = np.array((*ang_parse, *dist_parse))
            return parse

        if type == 'col':
            store = np.zeros((2, 3), dtype=int)
            for l in range(2):
                ang_parse = self.parse_coord(mus[l][0:2], num_deg, '2')
                dist_parse = self.parse_coord(mus[l][2], num_deg, '1')
                parse = np.array((*ang_parse, *dist_parse))
                store[l] = parse
            return store

        if type == 'jnt':
            store = np.zeros((2, 3, 3), dtype=int)
            for s in range(2):
                for l in range(3):
                    ang_parse = self.parse_coord(mus[s][l][0:2], num_deg, '2')
                    dist_parse = self.parse_coord(mus[s][l][2], num_deg, '1')
                    parse = np.array((*ang_parse, *dist_parse))
                    store[s][l] = parse
            return store

    # parse general variables with 1 or 2 coordinates
    def parse_coord(self, mus, num_deg, type):
        ptts = ptss(num_deg, num_deg)
        if type == '1':
            agn, ant = mus
            parse = ptts.parse(np.array((agn, 0)), np.array((ant, 0)), '1')
            return parse

        if type == '2':
            agn, ant = np.transpose(mus, (1, 0))
            parse = ptts.parse(agn, ant, '2')
            return parse

    # interchange rows in array
    def rev_row(self, x):
        out = np.zeros((2, 2, 2))
        for s in range(2):
            out[s][0] = x[s][1]
            out[s][1] = x[s][0]
        return out


    # form binocular map from fusing left and right retinal maps across depths
    def binoc_fuse(self, y, x, size):
        out = self.cfns.binoc_fuse(y, x, size)
        return out

    # filter binocular map with nearest neighbor kernel
    def binoc_map_trans(self, map, size):
        pre_out = self.cfns.binoc_map(map, size)
        post_out = 8 * self.thresh_fn(pre_out, 1 / 8)
        norm = np.sum(post_out)
        return post_out / (norm + 0.01)

    # form retinal map centered at foveae
    def gaze_map_trans(self, pt, x, size):
        out = self.cfns.gaze_map(pt, x, size)
        return out

    # check difference within [-1, 1]
    def compare(self, x, y):
        if abs(x[0] - y[0]) == 0 and abs(x[1] - y[1]) <= 1:
            return 1
        if abs(x[0] - y[0]) <= 1 and abs(x[1] - y[1]) == 0:
            return 1
        else:
            return 0

    # filter retinal map with gaussian kernel in the center
    def sc_map_cent(self, map, sigma, size):
        out = self.cfns.sc_map_cent(map, sigma, size)
        return out

    # filter retinal map with gaussian kernel in the surround
    def sc_map_surr(self, map, sigma, size):
        out = self.cfns.sc_map_surr(map, sigma, size)
        return out

    # gaussian gradient
    def gauss_gradient(self, q, p, j, i, sigma):
        r = np.sqrt((q-j)**2 + (p-i)**2)
        gradient = np.exp(-sigma * r **2)
        return gradient

    # gaussian kernel
    def gauss_kernel(self, j, i, sigma, size):
        kernel = np.zeros((size, size))
        bound = self.kernel_bound(j, i, size)
        norm = sum(self.gauss_gradient(q, p, j, i, sigma) for q in bound[0] for p in bound[1])
        for q in bound[0]:
            for p in bound[1]:
                kernel[q][p] = self.gauss_gradient(q, p, j, i, sigma)
        return kernel / norm

    # nearest neighbor kernel
    def indic_kernel(self, j, i, size):
        kernel = np.zeros((size, size))
        bound = self.kernel_bound(j, i, size)
        for q in bound[0]:
            for p in bound[1]:
                kernel[q][p] = 1/4
        return kernel

    # bound array index
    def kernel_bound(self, j, i, size):
        return range(max(j - 1, 0), min(j + 1, size)), \
               range(max(i - 1, 0), min(i + 1, size))



    # ____________________________________________________________________________________________________________
    # COLM, ARMS, LEGS Modules

    # generate random commands for the column, arms and legs during motor babbling
    def rand_prog(self, type):
        if type == 'col':
            # for head babbling and eye-head coordination
            rand = -0.4 + 0.8 * np.random.rand(2, 2)
            out = 0 * np.ones((2, 2, 2))

            for l in range(1):
                for m in range(2):
                    out[l][m][0] = rand[l][m]
                    out[l][m][1] = -out[l][m][0]

            return out

        if type == 'arm':
            # for eye-hand coordination
            #rand = 0.0 + 0.4 * np.random.rand(2, 3, 2)

            # for arm babbling
            rand = -0.4 + 0.8 * np.random.rand(2, 3, 2)
            out = 0 * np.ones((2, 3, 2, 2))

            # make commands across the limbs in the same direction as the commands in the shoulders
            sign = np.sign(rand[0][0][0]), np.sign(rand[1][0][0])
            rand = np.array((np.abs(rand[0]) * sign[0], np.abs(rand[1]) * sign[1]))

            for s in range(2):
                for l in range(3):
                    if l == 0:
                        for m in range(1):
                            out[s][l][m][0] = rand[s][l][m]
                            out[s][l][m][1] = -out[s][l][m][0]
                    if l == 1:
                        for m in range(1):
                            out[s][l][m][0] = rand[s][l][m] * self.indic_fn(rand[s][l][m])
                            out[s][l][m][1] = -out[s][l][m][0]
                    if l == 2:
                        for m in range(1):
                            out[s][l][m][0] = rand[s][l][m]
                            out[s][l][m][1] = -out[s][l][m][0]
            return out

        if type == 'leg':
            # for leg babbling
            rand = -0.4 + 0.8 * np.random.rand(2, 3, 2)
            out = 0 * np.ones((2, 3, 2, 2))

            # make commands across the limbs in the same direction as the commands in the hips
            sign = np.sign(rand[0][0][0]), np.sign(rand[1][0][0])
            rand = np.array((np.abs(rand[0]) * sign[0], np.abs(rand[1]) * sign[1]))

            for s in range(2):
                for l in range(3):
                    if l == 0:
                        for m in range(2):
                            out[s][l][m][0] = rand[s][l][m]
                            out[s][l][m][1] = -out[s][l][m][0]
            return out

        if type == 'step':
            # for eye-leg coordination
            rand = 0.0 + 0.4 * np.random.rand(2, 3, 2)
            out = 0 * np.ones((2, 3, 2, 2))

            for s in range(2):
                for l in range(3):
                    if l == 0:
                        for m in range(1):
                            out[s][l][m][0] = rand[s][l][m]
                            out[s][l][m][1] = -out[s][l][m][0]
            return out


        # special head positions to make it easier bring the hand/foot within the visual field
        if type == 'init':
            # for eye-hand/eye-leg coordination
            rand1 = -0.4 + 0.3 * np.random.rand()
            rand2 = -0.4 + 0.8 * np.random.rand()

            out = 0 * np.ones((2, 2, 2))

            for l in range(1):
                for m in range(2):
                    if m == 0:
                        out[l][m][1] = -rand1
                        out[l][m][0] = -out[l][m][1]
                    if m == 1:
                        out[l][m][0] = rand2
                        out[l][m][1] = -out[l][m][0]
            return out

    # preprocess commands
    def rand_fix(self, x, type):
        if type == 'arm':
            weight = np.array([((1, 1), (1, 1)), ((1, 0), (0, 0)), ((1, 1), (0, 0))])
            return x * weight

        if type == 'leg':
            weight = np.array([((1, 1), (1, 1)), ((0, 1), (0, 0)), ((1, 1), (0, 0))])
            return x * weight

        if type == 'step':
            weight = np.array([((1, 0), (1, 1)), ((0, 1), (0, 0)), ((1, 0), (0, 0))])
            return x * weight

        if type == 'rev':
            out = np.zeros((3, 2, 2))
            for l in range(3):
                for m in range(2):
                    for n in range(2):
                        out[l][m][n] = x[l][m][(n+1)%2]
            return out

    # compute difference btw agonist and antagonist for non-learning variables
    def diff_mus(self, x, type):
        if type == 'axial':
            mus = x - self.rev_mus(x, 'col')
            return mus

        if type == 'append':
            mus = x - self.rev_mus(x, 'jnt')
            return mus

    # compute muscle spindle input to rhythm generator
    def extract_spin(self, x):
        # given x has format (s, l, m, n)

        # only hip joints contribute
        left_hip, right_hip = x[0][0][0], x[1][0][0]
        rev_left, rev_right = self.rev_mus(left_hip, 'gen'), self.rev_mus(right_hip, 'gen')
        out = self.thresh_fn(left_hip - rev_left, 0) + self.thresh_fn(rev_right - right_hip, 0)

        return out

    # compute rhythm generator output to spinal cord
    def extract_rythm(self, x):
        # given x has format (s)
        out = np.zeros((2, 3, 2, 2))
        rev = self.rev_mus(x, 'gen')

        # only activate hip joints
        out[0][0][0] = self.thresh_fn(x - rev, 0)
        out[1][0][0] = self.thresh_fn(rev - x, 0)

        return out

    def sign(self, x, mode, type):
        if type == 'axial':
            if mode == 'agn':
                out = np.ones((2, 2, 2))
                for l in range(2):
                    for m in range(2):
                        out[l][m][0] = 1
                        out[l][m][1] = -1

                return np.abs(x) * out

            if mode == 'ant':
                out = np.ones((2, 2, 2))
                for l in range(2):
                    for m in range(2):
                        out[l][m][0] = -1
                        out[l][m][1] = 1

                return np.abs(x) * out

        if type == 'append':
            if mode == 'agn':
                out = np.ones((3, 2, 2))
                for l in range(3):
                    for m in range(2):
                        out[l][m][0] = 1
                        out[l][m][1] = -1

                return np.abs(x) * out

            if mode == 'ant':
                out = np.ones((3, 2, 2))
                for l in range(3):
                    for m in range(2):
                        out[l][m][0] = -1
                        out[l][m][1] = 1

                return np.abs(x) * out

    # switch btw agonist and antagonist
    def rev_mus(self, x, type):
        if type == 'eye' or type == 'col':
            mus = np.zeros((2, 2, 2))
            for s in range(2):
                for m in range(2):
                    for n in range(2):
                        mus[s][m][n] = x[s][m][(n + 1) % 2]
            return mus

        if type == 'jnt':
            mus = np.zeros((2, 3, 2, 2))
            for s in range(2):
                for l in range(3):
                    for m in range(2):
                        for n in range(2):
                            mus[s][l][m][n] = x[s][l][m][(n + 1) % 2]
            return mus

        if type == 'spt':
            mus = np.zeros((3, 2))
            for k in range(3):
                for n in range(2):
                    mus[k][n] = x[k][(n + 1) % 2]
            return mus

        if type == 'gen':
            mus = np.zeros(2)
            for n in range(2):
                mus[n] = x[(n + 1) % 2]
            return mus

    # compute net force btw agonist and antagonist
    def diff_force(self, x, type):
        if type == 'append':
            out = np.zeros((2, 3, 2))
            for s in range(2):
                for l in range(3):
                    for m in range(2):
                        out[s][l][m] = x[s][l][m][0] - x[s][l][m][1]
            return out

        if type == 'axial':
            out = np.zeros((2, 2))
            for l in range(2):
                for m in range(2):
                    out[l][m] = x[l][m][0] - x[l][m][1]
            return out

    # rearrange muscle components for the head and column
    def col_mus(self, x):
        out = np.zeros((2, 2, 2))
        for l in range(2):
            out[l][0][0] = (x[l][0][0] + x[l][0][1]) / 2
            out[l][0][1] = (x[l][1][0] + x[l][1][1]) / 2
            out[l][1][0] = (x[l][0][0] + x[l][1][0]) / 2
            out[l][1][1] = (x[l][0][1] + x[l][1][1]) / 2
        return out

    # rearrange muscle components for the arm
    def arm_mus(self, x):
        out = x
        return out

    # rearrange muscle components for the leg
    def leg_mus(self, x):
        out = x
        return out

    # compute joint receptor feedback
    def jnt_recept(self, x, type):
        out = np.zeros((2, 3, 2, 2))
        if type == 'arm':
            for s in range(2):
                if x[s][0][0] >= 0:
                    # compensation for horz shoul
                    out[s][0][1][0] = x[s][0][0]
                    out[s][0][1][1] = -out[s][0][1][0]

                    # compensation for elbow
                    out[s][1][0][0] = x[s][0][0]
                    out[s][1][0][1] = -out[s][1][0][0]

                if x[s][0][0] < 0:
                    # compensation for horz shoul
                    out[s][0][1][1] = -x[s][0][0]
                    out[s][0][1][0] = -out[s][0][1][1]

                    # compensation for elbow
                    out[s][1][0][1] = -x[s][0][0]
                    out[s][1][0][0] = -out[s][1][0][1]

                if x[s][1][0] >= 0:
                    # compensation for wrist
                    out[s][2][0][0] = x[s][1][0]
                    out[s][2][0][1] = -out[s][2][0][0]

                if x[s][1][0] < 0:
                    # compensation for wrist
                    out[s][2][0][1] = -x[s][1][0]
                    out[s][2][0][0] = -out[s][2][0][1]

            return out

        if type == 'leg':
            for s in range(2):
                if x[s][0][0] >= 0:
                    # compensation for horz hip
                    out[s][0][1][0] = x[s][0][0]
                    out[s][0][1][1] = -out[s][0][1][0]

                    # compensation for knee
                    out[s][1][0][0] = x[s][0][0]
                    out[s][1][0][1] = -out[s][1][0][0]

                    # compensation for ankle
                    pass

                if x[s][0][0] < 0:
                    # compensation for horz hip
                    out[s][0][1][1] = -x[s][0][0]
                    out[s][0][1][0] = -out[s][0][1][1]

                    # compensation for knee
                    out[s][1][0][1] = -x[s][0][0]
                    out[s][1][0][0] = -out[s][1][0][1]

                    # compensation for ankle
                    pass


                if x[s][1][0] >= 0:
                    # compensation for ankle
                    out[s][2][0][0] = x[s][1][0]
                    out[s][2][0][1] = -out[s][2][0][0]

                if x[s][1][0] < 0:
                    # compensation for ankle
                    out[s][2][0][1] = -x[s][1][0]
                    out[s][2][0][0] = -out[s][2][0][1]

            return out


    # compute net angle btw agonist and antagonist
    def diff_posn(self, x, type):
        if type == 'col':
            out = np.zeros((2, 2))
            for l in range(2):
                for m in range(2):
                    out[l][m] = x[l][m][0] - x[l][m][1]
            out = self.angle_bound(out, 'col')
            return out

        if type == 'arm':
            out = np.zeros((2, 3, 2))
            for s in range(2):
                for l in range(3):
                    for m in range(2):
                        out[s][l][m] = x[s][l][m][0] - x[s][l][m][1]
            out = self.angle_bound(out[0], 'arm'), self.angle_bound(out[1], 'arm')
            return out

        if type == 'leg':
            out = np.zeros((2, 3, 2))
            for s in range(2):
                for l in range(3):
                    for m in range(2):
                        out[s][l][m] = x[s][l][m][0] - x[s][l][m][1]
            out = self.angle_bound(out[0], 'leg'), self.angle_bound(out[1], 'leg')
            return out


    # convert into array format
    def arrform(self, x, type):
        if type == 'axial':
            out = np.zeros((2, 2, 2, 3))
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        out[l][m][n] = x[l][m][n]
            return out

        if type == 'append':
            out = np.zeros((2, 3, 2, 2, 3))
            for s in range(2):
                for l in range(3):
                    for m in range(2):
                        for n in range(2):
                            out[s][l][m][n] = x[s][l][m][n]
            return out



    # convert normalized activity into angle for limb variables
    def angle_bound(self, x, type):
        if type == 'col':
            z = x[0][0]
            neck_vert = 1.0 * np.pi / 6 * self.indic_fn(z) * np.sign(z) * z + \
                        1.0 * np.pi / 6 * self.indic_fn(-z) * np.sign(z) * (-z)

            z = x[0][1]
            neck_horz = 1.0 * np.pi / 6 * self.indic_fn(z) * np.sign(z) * z + \
                        1.0 * np.pi / 6 * self.indic_fn(-z) * np.sign(z) * (-z)

            z = x[1][0]
            pelvic_vert = np.pi / 90 * self.indic_fn(z) * np.sign(z) * z + \
                          np.pi / 90 * self.indic_fn(-z) * np.sign(z) * (-z)

            z = x[1][1]
            pelvic_horz = np.pi / 90 * self.indic_fn(z) * np.sign(z) * z + \
                          np.pi / 90 * self.indic_fn(-z) * np.sign(z) * (-z)

            return np.array([(neck_vert, neck_horz), (pelvic_vert, pelvic_horz)])

        if type == 'arm':
            z = x[0][0]
            shoul_vert = 1 * np.pi / 6 * self.indic_fn(z) * np.sign(z) * z + \
                    1 * np.pi / 6 * self.indic_fn(-z) * np.sign(z) * (-z)

            z = x[0][1]
            shoul_horz = 1 * np.pi / 6 * self.indic_fn(z) * np.sign(z) * z + \
                         1 * np.pi / 6 * self.indic_fn(-z) * np.sign(z) * (-z)

            z = x[1][0]
            elbow_vert = 1 * np.pi / 2 * self.indic_fn(z) * z

            z = x[2][0]
            wrist_vert = np.pi / 18 * self.indic_fn(z) * np.sign(z) * z + \
                         np.pi / 18 * self.indic_fn(-z) * np.sign(z) * (-z)

            return np.array([(shoul_vert, shoul_horz), (elbow_vert, 0), (wrist_vert, 0)])

        if type == 'leg':
            z = x[0][0]
            hip_vert = 1 * np.pi / 6 * self.indic_fn(z) * np.sign(z) * z + \
                       1 * np.pi / 6 * self.indic_fn(-z) * np.sign(z) * (-z)

            z = x[0][1]
            hip_horz = 1 * np.pi / 6 * self.indic_fn(z) * np.sign(z) * z + \
                       1 * np.pi / 6 * self.indic_fn(-z) * np.sign(z) * (-z)

            z = x[1][0]
            knee_vert = np.pi / 6 * self.indic_fn(-z) * np.sign(z) * (-z)

            z = x[2][0]
            ankle_vert = np.pi / 18 * self.indic_fn(z) * z + \
                         np.pi / 18 * self.indic_fn(-z) * np.sign(z) * (-z)

            return np.array([(hip_vert, hip_horz), (knee_vert, 0), (ankle_vert, 0)])



    # sample within some interval
    def rand_intv(self, a, b):
        return a + (b - a) * np.random.random()



    # parse limb variables
    def parse_append(self, mus, num_deg):
        ptss = ptssjnt(num_deg, num_deg)
        store = np.zeros((2, 3, 2), dtype=int)
        num = (2, 1, 1)
        for s in range(2):
            for l in range(3):
                agn, ant = np.transpose(mus[s][l], (1, 0))
                store[s][l] = ptss.parse(agn, ant, num[l])
                if num[l] == 1:
                    store[s][l][1] = int(num_deg / 2 - 1)
        return store

    # parse head/column variables
    def parse_axial(self, mus, num_deg):
        ptss = ptssjnt(num_deg, num_deg)
        store = np.zeros((2, 2), dtype=int)
        for l in range(2):
            agn, ant = np.transpose(mus[l], (1, 0))
            store[l] = ptss.parse(agn, ant, 2)
        return store

    # extract cartesian coordinates of virtual target position aligned with head center
    def head_extract(self, x):
        return x[0][2]

    # extract cartesian coordinates of hand position
    def hand_extract(self, x):
        left_limb, right_limb = x
        left_hand, right_hand = np.transpose(left_limb, (1, 0))[3], np.transpose(right_limb, (1, 0))[3]
        return left_hand, right_hand

    # extract cartesian coordinates of foot position
    def foot_extract(self, x):
        left_limb, right_limb = x
        left_foot, right_foot = np.transpose(left_limb, (1, 0))[3], np.transpose(right_limb, (1, 0))[3]
        return left_foot, right_foot

    # initialize eye position
    def eye_init(self):
        fovea = np.array([(0, 0), (0, 0)])
        return fovea

    # initialize head and column positions
    def column_init(self):
        neck = (0, 0)
        sacrum = (0, 0)
        return np.array((neck, sacrum))

    # initialize upper limb positions
    def uplimb_init(self):
        shoul = (0, 0)
        elbow = (0, 0)
        wrist = (0, 0)
        return np.array((shoul, elbow, wrist))

    # initialize lower limb positions
    def lowlimb_init(self):
        hip = (0, 0)
        knee = (0, 0)
        ankle = (0, 0)
        return np.array((hip, knee, ankle))

    # initialize present representation of lower limbs
    def foot_init(self):
        left = np.array((0.22, 0.03, 0.87))
        rev_left = 1 - left
        right = np.array((0.78, 0.03, 0.87))
        rev_right = 1 - right

        left_rep = np.transpose(np.array((left, rev_left)), (1, 0))
        left_limb = np.array([left_rep for l in range(3)])

        right_rep = np.transpose(np.array((right, rev_right)), (1, 0))
        right_limb = np.array([right_rep for l in range(3)])

        return np.array((left_limb, right_limb))

    # initialize present representation of upper limbs
    def hand_init(self):
        left = np.array((0, 0.04, 0.48))
        rev_left = 1 - left
        right = np.array((1, 0.04, 0.48))
        rev_right = 1 - right

        left_rep = np.transpose(np.array((left, rev_left)), (1, 0))
        left_limb = np.array([left_rep for l in range(3)])

        right_rep = np.transpose(np.array((right, rev_right)), (1, 0))
        right_limb = np.array([right_rep for l in range(3)])

        return np.array((left_limb, right_limb))

    # compute muscle lengths
    def mus_len(self, x, type):
        if type == 'axial':
            len = np.zeros((2, 2))
            for m in range(2):
                for n in range(2):
                    len[m][n] = np.linalg.norm(x[m][n])
            return len

        if type == 'append':
            len = np.zeros((3, 2, 2))
            for l in range(3):
                for m in range(2):
                    for n in range(2):
                        len[l][m][n] = np.linalg.norm(x[l][m][n])
            return len

        if type == 'single':
            return np.linalg.norm(x)

    # compute change in muscle lengths
    def mus_derv(self, old_x, new_x, h, type):
        if type == 'axial':
            derv = np.zeros((2, 2))
            for m in range(2):
                for n in range(2):
                    derv[m][n] = (new_x[m][n] - old_x[m][n]) / h
            return derv

        if type == 'append':
            derv = np.zeros((3, 2, 2))
            for l in range(3):
                for m in range(2):
                    for n in range(2):
                        derv[l][m][n] = (new_x[l][m][n] - old_x[l][m][n]) / h
            return derv

    # convert mode variables for stepping
    def conv_mode(self, x):
        if x == 0:
            return (0, 1)
        if x == 1:
            return (1, 0)


    # ____________________________________________________________________________________________________________

    # BMAP3D Module

    # convert to 2D polar coordinates
    def polar_tran(self, y, x):
        radius = np.sqrt(y ** 2 + x ** 2)

        if x == 0 and y == 0:
            theta00 = 0
            return (radius, theta00)

        elif x == 0 and y > 0:
            theta01 = np.pi / 2
            return (radius, theta01)

        elif x == 0 and y < 0:
            theta02 = -np.pi / 2
            return (radius, theta02)

        elif x > 0 and y >= 0:
            theta1 = np.arctan(y / x)
            return (radius, theta1)


        elif x > 0 and y <= 0:
            theta2 = np.arctan(y / x)
            return (radius, theta2)


        elif x < 0 and y <= 0:
            theta3 = -np.pi + np.arctan(y / x)
            return (radius, theta3)


        elif x < 0 and y >= 0:
            theta4 = np.pi + np.arctan(y / x)
            return (radius, theta4)

    # convert to cartesian coordinates from 2D polar coordinates
    def polar_to_cart_2D(self, rot, rad):
        return rad * np.array((np.cos(rot), np.sin(rot)))

    # convert to cartesian coordinates from 3D polar coordinates
    def polar_to_cart_3D(self, horz, vert, rad):
        return rad * np.array((np.cos(horz) * np.cos(vert), np.sin(horz) * np.cos(vert), np.sin(vert)))

    # convert coordinates by -90 degs horizontal rotation
    def latr_left(self, horz, vert, rad):
        return rad * np.array((np.sin(horz) * np.cos(vert), -np.cos(horz) * np.cos(vert), np.sin(vert)))

    # convert coordinates by +90 degs horizontal rotation
    def latr_right(self, horz, vert, rad):
        return rad * np.array((-np.sin(horz) * np.cos(vert), np.cos(horz) * np.cos(vert), np.sin(vert)))

    # convert coordinates by +0 degs horizontal rotation
    def latr_front(self, horz, vert, rad):
        return rad * np.array((np.cos(horz) * np.cos(vert), np.sin(horz) * np.cos(vert), np.sin(vert)))

    # convert coordinates by +180 degs horizontal rotation
    def latr_back(self, horz, vert, rad):
        return rad * np.array((-np.cos(horz) * np.cos(vert), -np.sin(horz) * np.cos(vert), np.sin(vert)))

    # convert coordinates by +90 degs vertical rotation
    def vert_up(self, horz, vert, rad):
        return rad * np.array((-np.cos(horz) * np.sin(vert), -np.sin(horz) * np.sin(vert), np.cos(vert)))

    # convert coordinates by -90 degs vertical rotation
    def vert_down(self, horz, vert, rad):
        return rad * np.array((np.cos(horz) * np.sin(vert), np.sin(horz) * np.sin(vert), -np.cos(vert)))

    # compute corner position in frontal plane
    def front_locate(self, ref, type, horz, vert, rad):
        if type == 'left':
            return ref + self.latr_left(horz, 0, rad)
        if type == 'right':
            return ref + self.latr_right(horz, 0, rad)
        if type == 'up':
            return ref + self.vert_up(horz, vert, rad)
        if type == 'down':
            return ref + self.vert_down(horz, vert, rad)

    # compute corner positions in frontal plane
    def front_plane(self, ref, horz, vert, rad):
        down = self.front_locate(ref, 'down', horz, vert, rad)
        up = self.front_locate(ref, 'up', horz, vert, rad)
        left_down = self.front_locate(down, 'left', horz, vert, rad)
        right_down = self.front_locate(down, 'right', horz, vert, rad)
        left_up = self.front_locate(up, 'left', horz, vert, rad)
        right_up = self.front_locate(up, 'right', horz, vert, rad)
        return np.array((left_down, right_down, left_up, right_up))

    # compute corner positions of head frame and left/right eye frames
    def head_plane(self, ref, horz, vert, rad):
        left_cent = ref + self.latr_left(horz, 0, 2 * rad)
        right_cent = ref + self.latr_right(horz, 0, 2 * rad)
        left_eye = self.front_plane(left_cent, horz, vert, rad)
        right_eye = self.front_plane(right_cent, horz, vert, rad)

        left_up = left_cent + self.latr_left(horz, 0, 2 * rad) + self.vert_up(horz, vert, 2 * rad)
        left_up = self.front_plane(left_up, horz, vert, rad)[2]

        left_down = left_cent + self.latr_left(horz, 0, 2 * rad) + self.vert_down(horz, vert, 2 * rad)
        left_down = self.front_plane(left_down, horz, vert, rad)[0]

        right_up = right_cent + self.latr_right(horz, 0, 2 * rad) + self.vert_up(horz, vert, 2 * rad)
        right_up = self.front_plane(right_up, horz, vert, rad)[3]

        right_down = right_cent + self.latr_right(horz, 0, 2 * rad) + self.vert_down(horz, vert, 2 * rad)
        right_down = self.front_plane(right_down, horz, vert, rad)[1]

        head = np.array((left_down, right_down, left_up, right_up))

        return head, left_eye, right_eye

    # compute corner positions in transversal plane
    def transv_locate(self, ref, type, horz, vert, rad):
        if type == 'front':
            return ref + self.latr_front(horz, vert, rad)
        if type == 'back':
            return ref + self.latr_back(horz, -vert, rad)
        if type == 'right':
            return ref + self.latr_right(horz, 0, rad)
        if type == 'left':
            return ref + self.latr_left(horz, 0, rad)

    # computer corner positions in transversal plane
    def transv_plane(self, ref, horz, vert, rad):
        front = self.transv_locate(ref, 'front', horz, vert, rad)
        back = self.transv_locate(ref, 'back', horz, vert, rad)
        right_front = self.transv_locate(front, 'right', horz, vert, rad)
        left_front = self.transv_locate(front, 'left', horz, vert, rad)
        right_back = self.transv_locate(back, 'right', horz, vert, rad)
        left_back = self.transv_locate(back, 'left', horz, vert, rad)
        return np.array((left_back, right_back, left_front, right_front))

    # compute line ends in front-back direction
    def sagit_line(self, ref, horz, vert, rad):
        front = self.transv_locate(ref, 'front', horz, vert, rad)
        back = self.transv_locate(ref, 'back', horz, vert, rad)
        return np.array((front, back))

    # compute line ends in up-down direction
    def front_line(self, ref, horz, vert, rad):
        up = self.front_locate(ref, 'up', horz, vert, rad)
        down = self.front_locate(ref, 'down', horz, vert, rad)
        return np.array((up, down))

    # compute line ends in right-left direction
    def transv_line(self, ref, horz, vert, rad):
        right = self.transv_locate(ref, 'right', horz, vert, rad)
        left = self.transv_locate(ref, 'left', horz, vert, rad)
        return np.array((right, left))

    # compute corner positions in a transversal plane rotated to right by 90 degs
    def right_locate(self, ref, type, horz, vert, rad):
        if type == 'front':
            return ref + self.latr_left(horz, 0, rad)
        if type == 'back':
            return ref + self.latr_right(horz, 0, rad)
        if type == 'right':
            return ref + self.latr_front(horz, vert, rad)
        if type == 'left':
            return ref + self.latr_back(horz, -vert, rad)

    # compute corner positions in a transversal plane rotated to right by 90 degs
    def right_plane(self, ref, horz, vert, rad):
        front = self.right_locate(ref, 'front', horz, vert, rad)
        back = self.right_locate(ref, 'back', horz, vert, rad)
        right_front = self.right_locate(front, 'right', horz, vert, rad)
        left_front = self.right_locate(front, 'left', horz, vert, rad)
        right_back = self.right_locate(back, 'right', horz, vert, rad)
        left_back = self.right_locate(back, 'left', horz, vert, rad)
        return np.array((left_back, right_back, left_front, right_front))

    # compute corner positions in a transversal plane rotated to left by -90 degs
    def left_locate(self, ref, type, horz, vert, rad):
        if type == 'front':
            return ref + self.latr_right(horz, 0, rad)
        if type == 'back':
            return ref + self.latr_left(horz, 0, rad)
        if type == 'right':
            return ref + self.latr_back(horz, -vert, rad)
        if type == 'left':
            return ref + self.latr_front(horz, vert, rad)

    # compute corner positions in a transversal plane rotated to left by -90 degs
    def left_plane(self, ref, horz, vert, rad):
        front = self.left_locate(ref, 'front', horz, vert, rad)
        back = self.left_locate(ref, 'back', horz, vert, rad)
        right_front = self.left_locate(front, 'right', horz, vert, rad)
        left_front = self.left_locate(front, 'left', horz, vert, rad)
        right_back = self.left_locate(back, 'right', horz, vert, rad)
        left_back = self.left_locate(back, 'left', horz, vert, rad)
        return np.array((left_back, right_back, left_front, right_front))

    # ____________________________________________________________________________________________________________




