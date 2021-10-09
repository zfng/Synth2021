import numpy as np
from FUNCS import FNS

# variable class for INPUT Module
class TargVar:
    def __init__(self, ret_size):
        self.ret_size = ret_size
        self.magnify = 7
        self.ocu_dist = 3
        self.ang_rang = np.radians((-45, 45))
        self.dist_rang = np.array((5, 50))

        self.target = np.zeros(3)
        self.target_polar = np.zeros((3))
        self.CoM = np.zeros(3)
        self.offset = np.array((0, 0, 1)) * (45 + 3 * self.magnify)

        self.head_ang = np.zeros(2)

        self.head_posn = np.zeros((3))
        self.hand_posn = np.zeros((2, 3))
        self.foot_posn = np.zeros((2, 3))

        self.head_data = np.zeros((3))
        self.hand_data = np.zeros((3))
        self.foot_data = np.zeros((3))


# method class for INPUT Module
class TargFun:
    def __init__(self, TargVar):
        self.Targ = TargVar
        self.FNS = FNS()
        self.ret_size = self.Targ.ret_size
        self.magnify = self.Targ.magnify
        self.ang_rang = self.Targ.ang_rang
        self.dist_rang = self.Targ.dist_rang
        self.ocu_dist = self.Targ.ocu_dist


    # compute retinal positions activated by an external target
    def world_targ_cpt(self, t, T):
        res = self.ret_size

        ang_rang = 1 * self.ang_rang
        dist_rang = 1.0 * self.dist_rang
        ran = ang_rang[1]
        head = self.Targ.head_ang

        if t % T == 0:

            # learn representation with eye and head movement


            # sample target for learning compensation for eye movements
            horz, vert, dist = self.rand_intv(1 * ang_rang[0], 1 * ang_rang[1]), \
                               self.rand_intv(1 * ang_rang[0], 1 * ang_rang[1]), \
                               self.rand_intv(dist_rang[0], 1 * dist_rang[1])
            self.Targ.target = (self.Targ.CoM + self.Targ.offset) + self.polar_to_cart_3D(horz, vert, dist)


            """
            # sample target for learning compensation for head movements
            horz, vert, dist = 0, 0, 9
            self.Targ.target = (self.Targ.CoM + self.Targ.offset) + self.polar_to_cart_3D(horz, vert, dist)
            """

            """
            # practice representation with eye and and head movement
            trial = t // T
            if trial % 2 == 0:
                # sample target for learning compensation for eye movements
                horz, vert, dist = self.rand_intv(0.5 * ang_rang[0], 0.5 * ang_rang[1]), \
                                   self.rand_intv(0.5 * ang_rang[0], 0.5 * ang_rang[1]), \
                                   self.rand_intv(dist_rang[0], 0.5 * dist_rang[1])
                self.Targ.target = (self.Targ.CoM + self.Targ.offset) + self.polar_to_cart_3D(horz, vert, dist)


            else:
                # sample target for learning compensation for head movements
                horz, vert, dist = 0, 0, 9
                self.Targ.target = (self.Targ.CoM + self.Targ.offset) + self.polar_to_cart_3D(horz, vert, dist)
            """

        targ = self.Targ.target
        delta = targ - (self.Targ.CoM + self.Targ.offset)

        norm3 = np.linalg.norm(delta * np.array((1, 1, 1)))

        # check within specified range of distance
        if dist_rang[0] <= norm3 and norm3 <= dist_rang[1]:

            dist = norm3
            norm2 = np.linalg.norm(delta * np.array((1, 1, 0)))

            # check within specified range of vertical angle and account for head position
            if ang_rang[0] <= np.arctan(delta[2] / (norm2 + 0.001)) - head[0] and np.arctan(
                    delta[2] / (norm2 + 0.001)) - head[0] <= ang_rang[1]:

                vert = np.arctan(delta[2] / (norm2 + 0.001)) - head[0]
                norm1 = np.linalg.norm(delta * np.array((1, 0, 0)))

                # check within specified range of horizontal angle and account for head position
                if ang_rang[0] <= np.arctan(delta[1] / (norm1 + 0.001)) - head[1] and np.arctan(
                        delta[1] / (norm1 + 0.001)) - head[1] <= ang_rang[1]:

                    horz = np.arctan(delta[1] / (norm1 + 0.001)) - head[1]

                    left_horz, left_vert = self.ret_actvn(horz, vert, dist)[0]
                    right_horz, right_vert = self.ret_actvn(horz, vert, dist)[1]

                    left_ret_x, left_ret_y = self.ret_coord(left_horz, left_vert, right_horz, right_vert, ran, res)[0]
                    right_ret_x, right_ret_y = self.ret_coord(left_horz, left_vert, right_horz, right_vert, ran, res)[1]

                    self.Targ.target_polar = np.array((np.degrees(horz + 1 * head[1]), np.degrees(vert + 1 * head[0]), dist))

                    return np.array([(left_ret_y, left_ret_x), (right_ret_y, right_ret_x)])


                else:
                    return np.array([(0, 0), (0, 0)])

            else:
                return np.array([(0, 0), (0, 0)])

        else:
            return np.array([(0, 0), (0, 0)])

    # compute retinal positions activated by an external target for eye-head coordination
    def world_head_cpt(self, t, T, sign):
        res = self.ret_size

        ang_rang = 1 * self.ang_rang
        dist_rang = 1 * self.dist_rang

        ran = ang_rang[1]
        head = self.Targ.head_ang

        if t % T == 0:
            horz, vert, dist = FNS().rand_intv(0.5 * ang_rang[1], 0.5 * ang_rang[1]), \
                               FNS().rand_intv(0.5 * ang_rang[0], 0.4 * ang_rang[0]), \
                               FNS().rand_intv(0.6 * dist_rang[1], 0.7 * dist_rang[1])
            self.Targ.target = (self.Targ.CoM + self.Targ.offset) + self.polar_to_cart_3D(abs(horz) * sign, vert, dist)

        targ = self.Targ.target
        delta = targ - (self.Targ.CoM + self.Targ.offset)

        norm3 = np.linalg.norm(delta * np.array((1, 1, 1)))

        # check within specified range of distance
        if dist_rang[0] <= norm3 and norm3 <= dist_rang[1]:

            dist = norm3
            norm2 = np.linalg.norm(delta * np.array((1, 1, 0)))

            # check within specified range of vertical angle and account for head position
            if ang_rang[0] <= np.arctan(delta[2] / (norm2 + 0.001)) - head[0] and np.arctan(
                    delta[2] / (norm2 + 0.001)) - head[0] <= ang_rang[1]:

                vert = np.arctan(delta[2] / (norm2 + 0.001)) - head[0]
                norm1 = np.linalg.norm(delta * np.array((1, 0, 0)))

                # check within specified range of horizontal angle and account for head position
                if ang_rang[0] <= np.arctan(delta[1] / (norm1 + 0.001)) - head[1] and np.arctan(
                        delta[1] / (norm1 + 0.001)) - head[1] <= ang_rang[1]:

                    horz = np.arctan(delta[1] / (norm1 + 0.001)) - head[1]

                    left_horz, left_vert = self.ret_actvn(horz, vert, dist)[0]
                    right_horz, right_vert = self.ret_actvn(horz, vert, dist)[1]

                    left_ret_x, left_ret_y = self.ret_coord(left_horz, left_vert, right_horz, right_vert, ran, res)[0]
                    right_ret_x, right_ret_y = self.ret_coord(left_horz, left_vert, right_horz, right_vert, ran, res)[1]

                    self.Targ.target_polar = np.array(
                        (np.degrees(horz + 1 * head[1]), np.degrees(vert + 1 * head[0]), dist))

                    return np.array([(left_ret_y, left_ret_x), (right_ret_y, right_ret_x)])


                else:
                    return np.array([(0, 0), (0, 0)])

            else:
                return np.array([(0, 0), (0, 0)])

        else:
            return np.array([(0, 0), (0, 0)])

    # compute retinal positions activated by an external target for eye-hand coordination
    def world_arm_cpt(self, t, T, sign):
        res = self.ret_size

        ang_rang = 2 * self.ang_rang
        dist_rang = 2.5 * self.dist_rang

        ran = ang_rang[1]
        head = self.Targ.head_ang

        if t % T == 0:
            horz, vert, dist = FNS().rand_intv(0.45 * ang_rang[1], 0.55 * ang_rang[1]), \
                               FNS().rand_intv(0.90 * ang_rang[0], 0.85 * ang_rang[0]), \
                               FNS().rand_intv(0.45 * dist_rang[1], 0.50 * dist_rang[1])
            self.Targ.target = (self.Targ.CoM + self.Targ.offset) + self.polar_to_cart_3D(abs(horz) * sign, vert, dist)

        targ = self.Targ.target
        delta = targ - (self.Targ.CoM + self.Targ.offset)

        norm3 = np.linalg.norm(delta * np.array((1, 1, 1)))

        # check within specified range of distance
        if dist_rang[0] <= norm3 and norm3 <= dist_rang[1]:

            dist = norm3
            norm2 = np.linalg.norm(delta * np.array((1, 1, 0)))

            # check within specified range of vertical angle and account for head position
            if ang_rang[0] <= np.arctan(delta[2] / (norm2 + 0.001)) - head[0] and np.arctan(
                    delta[2] / (norm2 + 0.001)) - head[0] <= ang_rang[1]:

                vert = np.arctan(delta[2] / (norm2 + 0.001)) - head[0]
                norm1 = np.linalg.norm(delta * np.array((1, 0, 0)))

                # check within specified range of horizontal angle and account for head position
                if ang_rang[0] <= np.arctan(delta[1] / (norm1 + 0.001)) - head[1] and np.arctan(
                        delta[1] / (norm1 + 0.001)) - head[1] <= ang_rang[1]:

                    horz = np.arctan(delta[1] / (norm1 + 0.001)) - head[1]

                    left_horz, left_vert = self.ret_actvn(horz, vert, dist)[0]
                    right_horz, right_vert = self.ret_actvn(horz, vert, dist)[1]

                    left_ret_x, left_ret_y = self.ret_coord(left_horz, left_vert, right_horz, right_vert, ran, res)[0]
                    right_ret_x, right_ret_y = self.ret_coord(left_horz, left_vert, right_horz, right_vert, ran, res)[1]

                    self.Targ.target_polar = np.array(
                        (np.degrees(horz + 1 * head[1]), np.degrees(vert + 1 * head[0]), dist))

                    return np.array([(left_ret_y, left_ret_x), (right_ret_y, right_ret_x)])


                else:
                    return np.array([(0, 0), (0, 0)])

            else:
                return np.array([(0, 0), (0, 0)])

        else:
            return np.array([(0, 0), (0, 0)])

    # compute retinal positions activated by an external target for eye-leg coordination
    def world_leg_cpt(self, t, T, sign):
        res = self.ret_size

        ang_rang = 2 * self.ang_rang
        dist_rang = 2.5 * self.dist_rang

        ran = ang_rang[1]
        head = self.Targ.head_ang

        if t % T == 0:
            horz, vert, dist = FNS().rand_intv(0.25 * ang_rang[1], 0.35 * ang_rang[1]), \
                               FNS().rand_intv(0.90 * ang_rang[0], 0.85 * ang_rang[0]), \
                               FNS().rand_intv(0.85 * dist_rang[1], 0.90 * dist_rang[1])
            self.Targ.target = (self.Targ.CoM + self.Targ.offset) + self.polar_to_cart_3D(abs(horz) * sign, vert, dist)

        targ = self.Targ.target
        delta = targ - (self.Targ.CoM + self.Targ.offset)

        norm3 = np.linalg.norm(delta * np.array((1, 1, 1)))

        # check within specified range of distance
        if dist_rang[0] <= norm3 and norm3 <= dist_rang[1]:

            dist = norm3
            norm2 = np.linalg.norm(delta * np.array((1, 1, 0)))

            # check within specified range of vertical angle and account for head position
            if ang_rang[0] <= np.arctan(delta[2] / (norm2 + 0.001)) - head[0] and np.arctan(
                    delta[2] / (norm2 + 0.001)) - head[0] <= ang_rang[1]:

                vert = np.arctan(delta[2] / (norm2 + 0.001)) - head[0]
                norm1 = np.linalg.norm(delta * np.array((1, 0, 0)))

                # check within specified range of horizontal angle and account for head position
                if ang_rang[0] <= np.arctan(delta[1] / (norm1 + 0.001)) - head[1] and np.arctan(
                        delta[1] / (norm1 + 0.001)) - head[1] <= ang_rang[1]:

                    horz = np.arctan(delta[1] / (norm1 + 0.001)) - head[1]

                    left_horz, left_vert = self.ret_actvn(horz, vert, dist)[0]
                    right_horz, right_vert = self.ret_actvn(horz, vert, dist)[1]

                    left_ret_x, left_ret_y = self.ret_coord(left_horz, left_vert, right_horz, right_vert, ran, res)[0]
                    right_ret_x, right_ret_y = self.ret_coord(left_horz, left_vert, right_horz, right_vert, ran, res)[1]

                    self.Targ.target_polar = np.array(
                        (np.degrees(horz + 1 * head[1]), np.degrees(vert + 1 * head[0]), dist))

                    return np.array([(left_ret_y, left_ret_x), (right_ret_y, right_ret_x)])


                else:
                    return np.array([(0, 0), (0, 0)])

            else:
                return np.array([(0, 0), (0, 0)])

        else:
            return np.array([(0, 0), (0, 0)])

    # compute retinal positions activated by virtual target aligned with head center during eye-hand coordination
    def self_head_cpt(self):
        res = self.ret_size
        ang_rang = 1 * self.ang_rang
        dist_rang = 1 * self.dist_rang
        ran = ang_rang[1]
        head = self.Targ.head_ang

        center = self.Targ.head_posn
        self.Targ.target = center

        targ = self.Targ.target
        delta = targ - (self.Targ.CoM + self.Targ.offset)

        norm3 = np.linalg.norm(delta * np.array((1, 1, 1)))

        # check within specified range of distance
        if dist_rang[0] <= norm3 and norm3 <= dist_rang[1]:

            dist = norm3
            norm2 = np.linalg.norm(delta * np.array((1, 1, 0)))

            # check within specified range of vertical angle and account for head position
            if ang_rang[0] <= np.arctan(delta[2] / (norm2 + 0.001)) - head[0] and np.arctan(
                    delta[2] / (norm2 + 0.001)) - head[0] <= ang_rang[1]:

                vert = np.arctan(delta[2] / (norm2 + 0.001)) - head[0]
                norm1 = np.linalg.norm(delta * np.array((1, 0, 0)))

                # check within specified range of horizontal angle and account for head position
                if ang_rang[0] <= np.arctan(delta[1] / (norm1 + 0.001)) - head[1] and np.arctan(
                        delta[1] / (norm1 + 0.001)) - head[1] <= ang_rang[1]:

                    horz = np.arctan(delta[1] / (norm1 + 0.001)) - head[1]

                    left_horz, left_vert = self.ret_actvn(horz, vert, dist)[0]
                    right_horz, right_vert = self.ret_actvn(horz, vert, dist)[1]

                    left_ret_x, left_ret_y = self.ret_coord(left_horz, left_vert, right_horz, right_vert, ran, res)[0]
                    right_ret_x, right_ret_y = self.ret_coord(left_horz, left_vert, right_horz, right_vert, ran, res)[1]

                    self.Targ.target_polar = np.array(
                        (np.degrees(horz + 1 * head[1]), np.degrees(vert + 1 * head[0]), dist))

                    return np.array([(left_ret_y, left_ret_x), (right_ret_y, right_ret_x)])

                else:
                    return np.array([(0, 0), (0, 0)])

            else:
                return np.array([(0, 0), (0, 0)])

        else:
            return np.array([(0, 0), (0, 0)])


    # compute retinal positions activated by the hand during eye-hand coordination and the activated
    # side is specified by mode
    def self_arm_cpt(self, mode):
        res = self.ret_size
        ang_rang = 2 * self.ang_rang
        dist_rang = 2.5 * self.dist_rang
        ran = ang_rang[1]
        head = self.Targ.head_ang
        left_hand, right_hand = self.Targ.hand_posn
        if mode == 0:
            self.Targ.target = left_hand
        else:
            self.Targ.target = right_hand

        targ = self.Targ.target
        delta = targ - (self.Targ.CoM + self.Targ.offset)

        norm3 = np.linalg.norm(delta * np.array((1, 1, 1)))

        # check within specified range of distance
        if dist_rang[0] <= norm3 and norm3 <= dist_rang[1]:

            dist = norm3
            norm2 = np.linalg.norm(delta * np.array((1, 1, 0)))

            # check within specified range of vertical angle and account for head position
            if ang_rang[0] <= np.arctan(delta[2] / (norm2 + 0.001)) - head[0] and np.arctan(
                    delta[2] / (norm2 + 0.001)) - head[0] <= ang_rang[1]:

                vert = np.arctan(delta[2] / (norm2 + 0.001)) - head[0]
                norm1 = np.linalg.norm(delta * np.array((1, 0, 0)))

                # check within specified range of horizontal angle and account for head position
                if ang_rang[0] <= np.arctan(delta[1] / (norm1 + 0.001)) - head[1] and np.arctan(
                        delta[1] / (norm1 + 0.001)) - head[1] <= ang_rang[1]:

                    horz = np.arctan(delta[1] / (norm1 + 0.001)) - head[1]

                    left_horz, left_vert = self.ret_actvn(horz, vert, dist)[0]
                    right_horz, right_vert = self.ret_actvn(horz, vert, dist)[1]

                    left_ret_x, left_ret_y = self.ret_coord(left_horz, left_vert, right_horz, right_vert, ran, res)[0]
                    right_ret_x, right_ret_y = self.ret_coord(left_horz, left_vert, right_horz, right_vert, ran, res)[1]

                    self.Targ.target_polar = np.array((np.degrees(horz + 1 * head[1]), np.degrees(vert + 1 * head[0]), dist))

                    return np.array([(left_ret_y, left_ret_x), (right_ret_y, right_ret_x)])

                else:
                    return np.array([(0, 0), (0, 0)])

            else:
                return np.array([(0, 0), (0, 0)])

        else:
            return np.array([(0, 0), (0, 0)])

    # compute retinal positions activated by the foot during eye-leg coordination and the forward
    # side is specified by mode
    def self_leg_cpt(self, mode):
        res = self.ret_size
        ang_rang = 2.0 * self.ang_rang
        dist_rang = 2.5 * self.dist_rang
        ran = ang_rang[1]
        head = self.Targ.head_ang
        left_foot, right_foot = self.Targ.foot_posn
        if mode == 0:
            self.Targ.target = left_foot
        else:
            self.Targ.target = right_foot

        targ = self.Targ.target
        delta = targ - (self.Targ.CoM + self.Targ.offset)

        norm3 = np.linalg.norm(delta * np.array((1, 1, 1)))

        # check within specified range of distance
        if dist_rang[0] <= norm3 and norm3 <= dist_rang[1]:

            dist = norm3
            norm2 = np.linalg.norm(delta * np.array((1, 1, 0)))

            # check within specified range of vertical angle and account for head position
            if ang_rang[0] <= np.arctan(delta[2] / (norm2 + 0.001)) - head[0] and np.arctan(
                    delta[2] / (norm2 + 0.001)) - head[0] <= ang_rang[1]:

                vert = np.arctan(delta[2] / (norm2 + 0.001)) - head[0]
                norm1 = np.linalg.norm(delta * np.array((1, 0, 0)))

                # check within specified range of horizontal angle and account for head position
                if ang_rang[0] <= np.arctan(delta[1] / (norm1 + 0.001)) - head[1] and np.arctan(
                        delta[1] / (norm1 + 0.001)) - head[1] <= ang_rang[1]:

                    horz = np.arctan(delta[1] / (norm1 + 0.001)) - head[1]

                    left_horz, left_vert = self.ret_actvn(horz, vert, dist)[0]
                    right_horz, right_vert = self.ret_actvn(horz, vert, dist)[1]

                    left_ret_x, left_ret_y = self.ret_coord(left_horz, left_vert, right_horz, right_vert, ran, res)[0]
                    right_ret_x, right_ret_y = self.ret_coord(left_horz, left_vert, right_horz, right_vert, ran, res)[1]

                    self.Targ.target_polar = np.array((np.degrees(horz + 1 * head[1]), np.degrees(vert + 1 * head[0]), dist))

                    return np.array([(left_ret_y, left_ret_x), (right_ret_y, right_ret_x)])

                else:
                    return np.array([(0, 0), (0, 0)])

            else:
                return np.array([(0, 0), (0, 0)])

        else:
            return np.array([(0, 0), (0, 0)])

    # track head position
    def trk_head(self):
        head = self.Targ.head_ang
        center = self.Targ.head_posn

        delta = center - (self.Targ.CoM + self.Targ.offset)

        norm3 = np.linalg.norm(delta * np.array((1, 1, 1)))
        dist = norm3

        norm2 = np.linalg.norm(delta * np.array((1, 1, 0)))
        vert = np.arctan(delta[2] / (norm2 + 0.001)) - head[0]

        norm1 = np.linalg.norm(delta * np.array((1, 0, 0)))
        horz = np.arctan(delta[1] / (norm1 + 0.001)) - head[1]

        self.Targ.head_data = np.array((np.degrees(horz + 1 * head[1]), np.degrees(vert + 1 * head[0]), dist))

    # track hand position of the activated side
    def trk_arm(self, mode):
        head = self.Targ.head_ang
        left_hand, right_hand = self.Targ.hand_posn
        if mode == 0:
            hand = left_hand
        else:
            hand = right_hand

        delta = hand - (self.Targ.CoM + self.Targ.offset)

        norm3 = np.linalg.norm(delta * np.array((1, 1, 1)))
        dist = norm3

        norm2 = np.linalg.norm(delta * np.array((1, 1, 0)))
        vert = np.arctan(delta[2] / (norm2 + 0.001)) - head[0]

        norm1 = np.linalg.norm(delta * np.array((1, 0, 0)))
        horz = np.arctan(delta[1] / (norm1 + 0.001)) - head[1]

        self.Targ.hand_data = np.array((np.degrees(horz + 1 * head[1]), np.degrees(vert + 1 * head[0]), dist))

    # track foot position of the forward side
    def trk_leg(self, mode):
        head = self.Targ.head_ang
        left_foot, right_foot = self.Targ.foot_posn
        if mode == 0:
            foot = left_foot
        else:
            foot = right_foot

        delta = foot - (self.Targ.CoM + self.Targ.offset)

        norm3 = np.linalg.norm(delta * np.array((1, 1, 1)))
        dist = norm3

        norm2 = np.linalg.norm(delta * np.array((1, 1, 0)))
        vert = np.arctan(delta[2] / (norm2 + 0.001)) - head[0]

        norm1 = np.linalg.norm(delta * np.array((1, 0, 0)))
        horz = np.arctan(delta[1] / (norm1 + 0.001)) - head[1]

        self.Targ.foot_data = np.array((np.degrees(horz + 1 * head[1]), np.degrees(vert + 1 * head[0]), dist))


    # compute induced angles by a visual target
    def ret_actvn(self, horz, vert, dist):
        left_horz = np.arctan((dist * np.sin(horz) + self.ocu_dist / 2) / (dist * np.cos(horz) + 0.001))

        left_val = (dist * np.sin(vert)) / (np.sqrt(
            (dist * np.sin(horz) + self.ocu_dist / 2) ** 2 + (dist * np.cos(horz)) ** 2) + 0.001)

        left_vert = np.arcsin(self.cutoff_fn(left_val, 1))

        right_horz = np.arctan((dist * np.sin(horz) - self.ocu_dist / 2) / (dist * np.cos(horz) + 0.001))

        right_val = (dist * np.sin(vert)) / (np.sqrt(
            (dist * np.sin(horz) - self.ocu_dist / 2) ** 2 + (dist * np.cos(horz)) ** 2) + 0.001)

        right_vert = np.arcsin(self.cutoff_fn(right_val, 1))

        return np.array([(left_horz, left_vert), (right_horz, right_vert)])

    # convert induced angles to corresponding retinal positions by a visual target
    def ret_coord(self, left_horz, left_vert, right_horz, right_vert, ran, res):
        left_ret_x = np.floor((left_horz / ran) * res) * FNS().indic_fn(left_horz) + np.ceil(
            (left_horz / ran) * res) * FNS().indic_fn(-left_horz)

        left_ret_y = np.floor((left_vert / ran) * res) * FNS().indic_fn(left_vert) + np.ceil(
            (left_vert / ran) * res) * FNS().indic_fn(-left_vert)

        right_ret_x = np.floor((right_horz / ran) * res) * FNS().indic_fn(right_horz) + np.ceil(
            (right_horz / ran) * res) * FNS().indic_fn(-right_horz)

        right_ret_y = np.floor((right_vert / ran) * res) * FNS().indic_fn(right_vert) + np.ceil(
            (right_vert / ran) * res) * FNS().indic_fn(-right_vert)

        return np.array([(left_ret_x, left_ret_y), (right_ret_x, right_ret_y)], dtype=int)

    # convert to cartesian coordinates from 3D polar coordinates
    def polar_to_cart_3D(self, horz, vert, dist):
        return np.array((dist * np.cos(horz) * np.cos(vert), dist * np.sin(horz) * np.cos(vert), dist * np.sin(vert)))

    # sample within some interval
    def rand_intv(self, a, b):
        return a + (b - a) * np.random.random()

    # process input within bound
    def cutoff_fn(self, x, thresh):
        if x >= -thresh and x <= thresh:
            return x
        elif x < -thresh:
            return -thresh
        elif x > thresh:
            return thresh