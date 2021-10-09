import numpy as np
import matplotlib.pyplot as plt
from FUNCS import FNS

# variable class for body frame module
class MapVar:
    def __init__(self, ax, limit, origin, ret_size):
        self.ax = ax
        self.origin = origin
        self.center = origin
        self.ret_size = ret_size
        self.trk_change = 0
        self.offset = 0

        self.ax.set_xlim(0, limit[0])
        self.ax.set_ylim(0, limit[1])
        self.ax.set_zlim(0, limit[2])

        # target variables
        self.target = np.zeros(3)
        self.estimate = np.zeros(3)
        self.targ_data = np.zeros((2, 2))
        self.targ, = self.ax.plot([], [], [], 'o', color='blue', markersize=6, label='veridical')
        self.targ_line, = self.ax.plot([], [], [], color='red', linestyle='dotted')
        self.left_line, = self.ax.plot([], [], [], color='blue', linestyle='dotted')
        self.right_line, = self.ax.plot([], [], [], color='blue', linestyle='dotted')
        self.cent_line, = self.ax.plot([], [], [], color='black', linestyle='dotted')

        # estimate variables
        self.est, = self.ax.plot([], [], [], 'o', color='red', markersize=6, label='estimate')
        self.left_est, = self.ax.plot([], [], [], color='red', linestyle='dotted')
        self.right_est, = self.ax.plot([], [], [], color='red', linestyle='dotted')

        # body frame variables
        self.head, = self.ax.plot([], [], [], color='black')
        self.head_cent, = self.ax.plot([], [], [], 'x', color='black', markersize=2.5)
        self.left_eye, = self.ax.plot([], [], [], color='black')
        self.right_eye, = self.ax.plot([], [], [], color='black')
        self.left_cent, = self.ax.plot([], [], [], 'x', color='black', markersize=2.5)
        self.right_cent, = self.ax.plot([], [], [], 'x', color='black', markersize=2.5)
        self.left_fov, = self.ax.plot([], [], [], 'o', color='red', markersize=2.5)
        self.right_fov, = self.ax.plot([], [], [], 'o', color='red', markersize=2.5)

        self.column, = self.ax.plot([], [], [], color='black')
        self.column_jnt, = self.ax.plot([], [], [], 'o', color='red', markersize=2.5)
        self.pectoral, = self.ax.plot([], [], [], color='black')
        self.pelvic, = self.ax.plot([], [], [], color='black')
        self.CoM, = self.ax.plot([], [], [], 'x', color='blue', markersize=2.5)

        self.left_uplimb, = self.ax.plot([], [], [], color='black')
        self.left_uplimb_jnt, = self.ax.plot([], [], [], 'o', color='red', markersize=2.5)
        self.right_uplimb, = self.ax.plot([], [], [], color='black')
        self.right_uplimb_jnt, = self.ax.plot([], [], [], 'o', color='red', markersize=2.5)

        self.left_lowlimb, = self.ax.plot([], [], [], color='black')
        self.left_lowlimb_jnt, = self.ax.plot([], [], [], 'o', color='red', markersize=2.5)
        self.right_lowlimb, = self.ax.plot([], [], [], color='black')
        self.right_lowlimb_jnt, = self.ax.plot([], [], [], 'o', color='red', markersize=2.5)

        # muscles variables
        self.base_frame, = self.ax.plot([], [], [], color='black')
        self.thorax_frame, = self.ax.plot([], [], [], color='black')
        self.lumbar_frame, = self.ax.plot([], [], [], color='black')
        self.pect_frame, = self.ax.plot([], [], [], color='black')
        self.humr_frame, = self.ax.plot([], [], [], color='black')
        self.shoul_frame, = self.ax.plot([], [], [], color='black')
        self.elbow_frame, = self.ax.plot([], [], [], color='black')
        self.pelv_frame, = self.ax.plot([], [], [], color='black')
        self.femr_frame, = self.ax.plot([], [], [], color='black')
        self.hip_frame, = self.ax.plot([], [], [], color='black')
        self.knee_frame, = self.ax.plot([], [], [], color='black')

        self.left_neck_mus, = self.ax.plot([], [], [], color='red')
        self.right_neck_mus, = self.ax.plot([], [], [], color='red')
        self.left_trunk_mus, = self.ax.plot([], [], [], color='red')
        self.right_trunk_mus, = self.ax.plot([], [], [], color='red')
        self.shoul_horz, = self.ax.plot([], [], [], color='red')
        self.shoul_vert, = self.ax.plot([], [], [], color='red')
        self.elbow_vert, = self.ax.plot([], [], [], color='red')
        self.wrist_vert, = self.ax.plot([], [], [], color='red')
        self.hip_horz, = self.ax.plot([], [], [], color='red')
        self.hip_vert, = self.ax.plot([], [], [], color='red')
        self.knee_vert, = self.ax.plot([], [], [], color='red')
        self.ankle_vert, = self.ax.plot([], [], [], color='red')

        self.column, = self.ax.plot([], [], [], color='black')
        self.left_limb, = self.ax.plot([], [], [], color='black')
        self.right_limb, = self.ax.plot([], [], [], color='black')
        self.left_joints, = self.ax.plot([], [], [], 'o', color='red', markersize=2.5)
        self.left_inserts, = self.ax.plot([], [], [], 'o', color='blue', markersize=2.5)
        self.right_joints, = self.ax.plot([], [], [], 'o', color='red', markersize=2.5)
        self.right_inserts, = self.ax.plot([], [], [], 'o', color='blue', markersize=2.5)

        # external force variables
        self.fof = np.zeros((2, 3, 2))
        self.grf = np.zeros((2, 3, 2))

# method class for body frame module
class MapFun:
    def __init__(self, eye_data, axial_data, append_data, MapVar):
        self.MapVar = MapVar
        self.FNS = FNS()
        self.origin = MapVar.origin
        self.ret_size = self.MapVar.ret_size
        self.magnify = 7
        self.ang_rang = np.radians((-45, 45))
        self.dist_rang = np.array((5, 50))
        self.default = -5

        # initialize eye positions and joint angles
        self.eye_rot = FNS().eye_init()
        self.neck_rot, self.trunk_rot = FNS().column_init()
        self.uplimb_rot = (FNS().uplimb_init(), FNS().uplimb_init())
        self.lowlimb_rot = (FNS().lowlimb_init(), FNS().lowlimb_init())

        # updated eye positions and joint angles
        self.eye_data = eye_data
        self.axial_data = axial_data
        self.append_data = append_data

    # draw lines of sight from target to eyes
    def targ_plt(self, head_cent, head_ahead, left_targ, right_targ):
        targ = self.MapVar.target
        targ_data = self.MapVar.targ_data

        self.MapVar.targ.set_data(targ[0], targ[1])
        self.MapVar.targ.set_3d_properties(targ[2])

        if np.array_equal(targ_data, np.zeros((2, 2))) != True:
            #targ_line = np.transpose(np.array((targ, head_cent)), (1, 0))
            #self.MapVar.targ_line.set_data(targ_line[0], targ_line[1])
            #self.MapVar.targ_line.set_3d_properties(targ_line[2])

            left_line = np.transpose(np.array((targ, left_targ)), (1, 0))
            self.MapVar.left_line.set_data(left_line[0], left_line[1])
            self.MapVar.left_line.set_3d_properties(left_line[2])

            right_line = np.transpose(np.array((targ, right_targ)), (1, 0))
            self.MapVar.right_line.set_data(right_line[0], right_line[1])
            self.MapVar.right_line.set_3d_properties(right_line[2])

            #cent_line = np.transpose(np.array((head_ahead, head_cent)), (1, 0))
            #self.MapVar.cent_line.set_data(cent_line[0], cent_line[1])
            #self.MapVar.cent_line.set_3d_properties(cent_line[2])

    # draw lines of sight from estimate to eyes
    def est_plt(self, est, left_fov, right_fov):
        self.MapVar.est.set_data(est[0], est[1])
        self.MapVar.est.set_3d_properties(est[2])

        left_est = np.transpose(np.array((est, left_fov)), (1, 0))
        self.MapVar.left_est.set_data(left_est[0], left_est[1])
        self.MapVar.left_est.set_3d_properties(left_est[2])

        right_est = np.transpose(np.array((est, right_fov)), (1, 0))
        self.MapVar.right_est.set_data(right_est[0], right_est[1])
        self.MapVar.right_est.set_3d_properties(right_est[2])

    # compute head and eye positions in the body frame and do not update if shift=0 indicates the feet are
    # driven into ground
    def head_cpt(self, shift):
        FNS = self.FNS
        magn = self.magnify
        size = self.ret_size
        left_targ_hit, right_targ_hit = self.MapVar.targ_data

        self.eye_rot = self.eye_data
        left_eye_rot, right_eye_rot = self.eye_rot

        if shift == 0:
            neck_rot_vert, neck_rot_horz = self.neck_rot
            truk_rot_vert, truk_rot_horz = self.trunk_rot

        else:
            self.neck_rot, self.trunk_rot = self.axial_data
            neck_rot_vert, neck_rot_horz = self.neck_rot
            truk_rot_vert, truk_rot_horz = self.trunk_rot

        column = self.column_cpt(shift)[0]

        base = np.array((column[0][0], column[1][0], column[2][0]))


        head_cent = base + FNS.vert_up(truk_rot_horz + neck_rot_horz, truk_rot_vert + neck_rot_vert, 3 * magn)

        head_ahead = head_cent + FNS.latr_front(truk_rot_horz + neck_rot_horz, truk_rot_vert + neck_rot_vert, 20)

        left_cent = head_cent + FNS.latr_left(truk_rot_horz + neck_rot_horz, 0, 2 * magn)
        right_cent = head_cent + FNS.latr_right(truk_rot_horz + neck_rot_horz, 0, 2 * magn)


        left_rad_est, left_ang_est = FNS.polar_tran((magn / size) * left_eye_rot[0], (magn / size) * left_eye_rot[1])
        left_fov = left_cent + FNS.latr_right(truk_rot_horz + neck_rot_horz, left_ang_est, left_rad_est)

        right_rad_est, right_ang_est = FNS.polar_tran((magn / size) * right_eye_rot[0], (magn / size) * right_eye_rot[1])
        right_fov = right_cent + FNS.latr_right(truk_rot_horz + neck_rot_horz, right_ang_est, right_rad_est)

        left_rad_verd, left_ang_verd = FNS.polar_tran((magn / size) * left_targ_hit[0], (magn / size) * left_targ_hit[1])
        left_targ = left_cent + FNS.latr_right(truk_rot_horz + neck_rot_horz, left_ang_verd, left_rad_verd)

        right_rad_verd, right_ang_verd = FNS.polar_tran((magn / size) * right_targ_hit[0], (magn / size) * right_targ_hit[1])
        right_targ = right_cent + FNS.latr_right(truk_rot_horz + neck_rot_horz, right_ang_verd, right_rad_verd)

        head, left_eye, right_eye = FNS.head_plane(head_cent, truk_rot_horz + neck_rot_horz, truk_rot_vert + neck_rot_vert, magn)

        head = np.transpose(np.array((head[0], head[1], head[3], head[2], head[0])), (1, 0))
        left_eye = np.transpose(np.array((left_eye[0], left_eye[1], left_eye[3], left_eye[2], left_eye[0])), (1, 0))
        right_eye = np.transpose(np.array((right_eye[0], right_eye[1], right_eye[3], right_eye[2], right_eye[0])), (1, 0))

        return (head_cent, head, head_ahead), (left_eye, right_eye), (left_cent, right_cent), \
               (left_fov, right_fov), (left_targ, right_targ)

    # draw head and eye positions
    def head_plt(self, head_cent, head, left_eye, right_eye, left_cent, right_cent, left_fov, right_fov):
        self.MapVar.head.set_data(head[0], head[1])
        self.MapVar.head.set_3d_properties(head[2])

        self.MapVar.head_cent.set_data(head_cent[0], head_cent[1])
        self.MapVar.head_cent.set_3d_properties(head_cent[2])

        self.MapVar.left_eye.set_data(left_eye[0], left_eye[1])
        self.MapVar.left_eye.set_3d_properties(left_eye[2])

        self.MapVar.left_cent.set_data(left_cent[0], left_cent[1])
        self.MapVar.left_cent.set_3d_properties(left_cent[2])

        self.MapVar.right_eye.set_data(right_eye[0], right_eye[1])
        self.MapVar.right_eye.set_3d_properties(right_eye[2])

        self.MapVar.right_cent.set_data(right_cent[0], right_cent[1])
        self.MapVar.right_cent.set_3d_properties(right_cent[2])

        self.MapVar.left_fov.set_data(left_fov[0], left_fov[1])
        self.MapVar.left_fov.set_3d_properties(left_fov[2])

        self.MapVar.right_fov.set_data(right_fov[0], right_fov[1])
        self.MapVar.right_fov.set_3d_properties(right_fov[2])


    # compute position of center of mass due to column and/or leg movements and mode=(0, 1) indicates left leg
    # swing and right leg stance and mode=(1, 0) the reverse situation
    def CoM_shift(self, mode):
        FNS = self.FNS
        origin = self.MapVar.origin
        dep = self.default

        truk_rot_vert, truk_rot_horz = self.axial_data[1]

        (left_hip_rot_vert, left_hip_rot_horz), (left_knee_rot_vert, left_knee_rot_horz), \
        (left_ankle_rot_vert, left_ankle_rot_horz) = self.append_data[1][0]

        (right_hip_rot_vert, right_hip_rot_horz), (right_knee_rot_vert, right_knee_rot_horz), \
        (right_ankle_rot_vert, right_ankle_rot_horz) = self.append_data[1][1]

        # shift of CoM due to column movement
        shift_col = FNS.vert_up(0, 0, 10) - FNS.vert_up(truk_rot_horz, truk_rot_vert, 10)

        if mode == (0, 1):

            # shift of CoM due to forward left leg movement
            shift_limb = FNS.vert_up(0, 0, 35) - FNS.vert_up(right_hip_rot_horz, right_hip_rot_vert, 20) - \
                         FNS.vert_up(right_hip_rot_horz, right_hip_rot_vert + right_knee_rot_vert, 15)

            shift = shift_col + shift_limb

            # check if left foot is driven into ground
            left_foot, right_foot = self.lowlimb_tst(shift)
            if left_foot[2] < dep:
                shift = np.zeros(3)
                self.MapVar.center = origin - shift
                return 0

            else:
                shift = shift * np.array((1, -1, 1))
                self.MapVar.offset = shift * self.MapVar.trk_change + self.MapVar.offset * (1 - self.MapVar.trk_change)

                # update CoM position
                self.MapVar.center = origin - shift + self.MapVar.offset
                return 1


        if mode == (1, 0):

            # shift of CoM due to forward right leg movement
            shift_limb = FNS.vert_up(0, 0, 35) - FNS.vert_up(left_hip_rot_horz, left_hip_rot_vert, 20) - \
                         FNS.vert_up(left_hip_rot_horz, left_hip_rot_vert + left_knee_rot_vert, 15)

            shift = shift_col + shift_limb

            # check if right foot is driven into ground
            left_foot, right_foot = self.lowlimb_tst(shift)
            if right_foot[2] < dep:
                shift = np.zeros(3)
                self.MapVar.center = origin - shift
                return 0

            else:
                shift = shift * np.array((1, 1, 1))
                self.MapVar.offset = shift * self.MapVar.trk_change + self.MapVar.offset * (1 - self.MapVar.trk_change)

                # update CoM position
                self.MapVar.center = origin - shift + self.MapVar.offset
                return 1

    # compute positions of base of head, cervic (neck), thorax (for pectoral girdle), lumbar (CoM), and sacrum
    # (pelvic and for pelvic girdle)
    def column_cpt(self, shift):
        FNS = self.FNS

        if shift == 0:
            neck_rot_vert, neck_rot_horz = self.neck_rot
            truk_rot_vert, truk_rot_horz = self.trunk_rot

        else:
            self.neck_rot, self.trunk_rot = self.axial_data
            neck_rot_vert, neck_rot_horz = self.neck_rot
            truk_rot_vert, truk_rot_horz = self.trunk_rot

        center = self.MapVar.center

        lumbar = center
        sacrum = lumbar - FNS.vert_up(truk_rot_horz, truk_rot_vert, 10)
        thorax = lumbar + FNS.vert_up(truk_rot_horz, truk_rot_vert, 30)
        cervic = thorax + FNS.vert_up(truk_rot_horz, truk_rot_vert, 10)
        base = cervic + FNS.vert_up(truk_rot_horz + neck_rot_horz, truk_rot_vert + neck_rot_vert, 5)

        left_pectoral = thorax + FNS.latr_left(truk_rot_horz, 0, 10)
        right_pectoral = thorax + FNS.latr_right(truk_rot_horz, 0, 10)

        left_pelvic = sacrum + FNS.latr_left(0, 0, 5)
        right_pelvic = sacrum + FNS.latr_right(0, 0, 5)

        column = np.transpose(np.array((base, cervic, thorax, lumbar, sacrum)), (1, 0))
        pectoral = np.transpose(np.array((left_pectoral, thorax, right_pectoral)), (1, 0))
        pelvic = np.transpose(np.array((left_pelvic, sacrum, right_pelvic)), (1, 0))

        return column, pectoral, pelvic

    # draw positions of column segments
    def column_plt(self, column, pectoral, pelvic):
        self.MapVar.column.set_data(column[0], column[1])
        self.MapVar.column.set_3d_properties(column[2])

        self.MapVar.pectoral.set_data(pectoral[0], pectoral[1])
        self.MapVar.pectoral.set_3d_properties(pectoral[2])

        self.MapVar.pelvic.set_data(pelvic[0], pelvic[1])
        self.MapVar.pelvic.set_3d_properties(pelvic[2])

        cervic = (column[0][1], column[1][1], column[2][1])
        sacrum = (column[0][4], column[1][4], column[2][4])
        CoM = (column[0][3], column[1][3], column[2][3])
        column_jnt = np.transpose(np.array((cervic, sacrum)), (1, 0))
        self.MapVar.column_jnt.set_data(column_jnt[0], column_jnt[1])
        self.MapVar.column_jnt.set_3d_properties(column_jnt[2])

        self.MapVar.CoM.set_data(CoM[0], CoM[1])
        self.MapVar.CoM.set_3d_properties(CoM[2])

    # compute positions of shoulders elbows and wrists of upper limbs
    def uplimb_cpt(self, shift):
        FNS = self.FNS

        pectoral = self.column_cpt(shift)[1]
        left_shoulder = np.array((pectoral[0][0], pectoral[1][0], pectoral[2][0]))
        right_shoulder = np.array((pectoral[0][2], pectoral[1][2], pectoral[2][2]))

        if shift == 0:
            (left_shoul_rot_vert, left_shoul_rot_horz), (left_elbow_rot_vert, left_elbow_rot_horz), \
            (left_wrist_rot_vert, left_wrist_rot_horz) = self.uplimb_rot[0]

            (right_shoul_rot_vert, right_shoul_rot_horz), (right_elbow_rot_vert, right_elbow_rot_horz), \
            (right_wrist_rot_vert, right_wrist_rot_horz) = self.uplimb_rot[1]

        else:
            self.uplimb_rot = self.append_data[0]

            (left_shoul_rot_vert, left_shoul_rot_horz), (left_elbow_rot_vert, left_elbow_rot_horz), \
            (left_wrist_rot_vert, left_wrist_rot_horz) = self.uplimb_rot[0]

            (right_shoul_rot_vert, right_shoul_rot_horz), (right_elbow_rot_vert, right_elbow_rot_horz), \
            (right_wrist_rot_vert, right_wrist_rot_horz) = self.uplimb_rot[1]

        left_elbow = left_shoulder + FNS.vert_down(left_shoul_rot_horz, left_shoul_rot_vert, 15)

        left_wrist = left_elbow + FNS.vert_down(left_shoul_rot_horz, left_shoul_rot_vert + left_elbow_rot_vert, 10)

        left_hand = left_wrist + FNS.vert_down(left_shoul_rot_horz, left_shoul_rot_vert + left_elbow_rot_vert +
                                               left_wrist_rot_vert, 5)

        right_elbow = right_shoulder + FNS.vert_down(right_shoul_rot_horz, right_shoul_rot_vert, 15)

        right_wrist = right_elbow + FNS.vert_down(right_shoul_rot_horz, right_shoul_rot_vert + right_elbow_rot_vert, 10)

        right_hand = right_wrist + FNS.vert_down(right_shoul_rot_horz, right_shoul_rot_vert + right_elbow_rot_vert +
                                                 right_wrist_rot_vert, 5)

        left_limb = np.transpose(np.array((left_shoulder, left_elbow, left_wrist, left_hand)), (1, 0))
        right_limb = np.transpose(np.array((right_shoulder, right_elbow, right_wrist, right_hand)), (1, 0))

        return left_limb, right_limb

    # draw positions of upper limbs
    def uplimb_plt(self, left_uplimb, right_uplimb):
        self.MapVar.left_uplimb.set_data(left_uplimb[0], left_uplimb[1])
        self.MapVar.left_uplimb.set_3d_properties(left_uplimb[2])

        left_shoul = (left_uplimb[0][0], left_uplimb[1][0], left_uplimb[2][0])
        left_elbow = (left_uplimb[0][1], left_uplimb[1][1], left_uplimb[2][1])
        left_wrist = (left_uplimb[0][2], left_uplimb[1][2], left_uplimb[2][2])
        left_uplimb_jnt = np.transpose(np.array((left_shoul, left_elbow, left_wrist)), (1, 0))
        self.MapVar.left_uplimb_jnt.set_data(left_uplimb_jnt[0], left_uplimb_jnt[1])
        self.MapVar.left_uplimb_jnt.set_3d_properties(left_uplimb_jnt[2])

        self.MapVar.right_uplimb.set_data(right_uplimb[0], right_uplimb[1])
        self.MapVar.right_uplimb.set_3d_properties(right_uplimb[2])

        right_shoul = (right_uplimb[0][0], right_uplimb[1][0], right_uplimb[2][0])
        right_elbow = (right_uplimb[0][1], right_uplimb[1][1], right_uplimb[2][1])
        right_wrist = (right_uplimb[0][2], right_uplimb[1][2], right_uplimb[2][2])
        right_uplimb_jnt = np.transpose(np.array((right_shoul, right_elbow, right_wrist)), (1, 0))
        self.MapVar.right_uplimb_jnt.set_data(right_uplimb_jnt[0], right_uplimb_jnt[1])
        self.MapVar.right_uplimb_jnt.set_3d_properties(right_uplimb_jnt[2])

    # compute positions of hips, knees and ankles of lower limbs
    def lowlimb_cpt(self, shift):
        FNS = self.FNS
        pelvic = self.column_cpt(shift)[2]

        left_hip = np.array((pelvic[0][0], pelvic[1][0], pelvic[2][0]))
        right_hip = np.array((pelvic[0][2], pelvic[1][2], pelvic[2][2]))

        if shift == 0:
            (left_hip_rot_vert, left_hip_rot_horz), (left_knee_rot_vert, left_knee_rot_horz), \
            (left_ankle_rot_vert, left_ankle_rot_horz) = self.lowlimb_rot[0]

            (right_hip_rot_vert, right_hip_rot_horz), (right_knee_rot_vert, right_knee_rot_horz), \
            (right_ankle_rot_vert, right_ankle_rot_horz) = self.lowlimb_rot[1]

        else:
            self.lowlimb_rot = self.append_data[1]

            (left_hip_rot_vert, left_hip_rot_horz), (left_knee_rot_vert, left_knee_rot_horz), \
            (left_ankle_rot_vert, left_ankle_rot_horz) = self.lowlimb_rot[0]

            (right_hip_rot_vert, right_hip_rot_horz), (right_knee_rot_vert, right_knee_rot_horz), \
            (right_ankle_rot_vert, right_ankle_rot_horz) = self.lowlimb_rot[1]

        left_knee = left_hip + FNS.vert_down(left_hip_rot_horz, left_hip_rot_vert, 20)

        left_ankle = left_knee + FNS.vert_down(left_hip_rot_horz, left_hip_rot_vert + left_knee_rot_vert, 15)

        left_foot = left_ankle + FNS.vert_down(left_hip_rot_horz, left_hip_rot_vert + left_knee_rot_vert +
                                               left_ankle_rot_vert + np.pi / 2, 5)

        left_limb = np.transpose(np.array((left_hip, left_knee, left_ankle, left_foot)), (1, 0))

        right_knee = right_hip + FNS.vert_down(right_hip_rot_horz, right_hip_rot_vert, 20)

        right_ankle = right_knee + FNS.vert_down(right_hip_rot_horz, right_hip_rot_vert + right_knee_rot_vert, 15)

        right_foot = right_ankle + FNS.vert_down(right_hip_rot_horz, right_hip_rot_vert + right_knee_rot_vert +
                                                 right_ankle_rot_vert + np.pi / 2, 5)

        right_limb = np.transpose(np.array((right_hip, right_knee, right_ankle, right_foot)), (1, 0))

        return left_limb, right_limb

    # draw positions of lower limbs
    def lowlimb_plt(self, left_lowlimb, right_lowlimb):
        self.MapVar.left_lowlimb.set_data(left_lowlimb[0], left_lowlimb[1])
        self.MapVar.left_lowlimb.set_3d_properties(left_lowlimb[2])

        left_hip = (left_lowlimb[0][0], left_lowlimb[1][0], left_lowlimb[2][0])
        left_knee = (left_lowlimb[0][1], left_lowlimb[1][1], left_lowlimb[2][1])
        left_ankle = (left_lowlimb[0][2], left_lowlimb[1][2], left_lowlimb[2][2])
        left_lowlimb_jnt = np.transpose(np.array((left_hip, left_knee, left_ankle)), (1, 0))
        self.MapVar.left_lowlimb_jnt.set_data(left_lowlimb_jnt[0], left_lowlimb_jnt[1])
        self.MapVar.left_lowlimb_jnt.set_3d_properties(left_lowlimb_jnt[2])

        self.MapVar.right_lowlimb.set_data(right_lowlimb[0], right_lowlimb[1])
        self.MapVar.right_lowlimb.set_3d_properties(right_lowlimb[2])

        right_hip = (right_lowlimb[0][0], right_lowlimb[1][0], right_lowlimb[2][0])
        right_knee = (right_lowlimb[0][1], right_lowlimb[1][1], right_lowlimb[2][1])
        right_ankle = (right_lowlimb[0][2], right_lowlimb[1][2], right_lowlimb[2][2])
        right_lowlimb_jnt = np.transpose(np.array((right_hip, right_knee, right_ankle)), (1, 0))
        self.MapVar.right_lowlimb_jnt.set_data(right_lowlimb_jnt[0], right_lowlimb_jnt[1])
        self.MapVar.right_lowlimb_jnt.set_3d_properties(right_lowlimb_jnt[2])

    # test if shift of CoM would cause either feet into ground
    def lowlimb_tst(self, shift):
        FNS = self.FNS

        neck_rot_vert, neck_rot_horz = self.axial_data[0]
        truk_rot_vert, truk_rot_horz = self.axial_data[1]

        center = self.MapVar.origin - shift

        sacrum = center - FNS.vert_up(truk_rot_horz, truk_rot_vert, 10)
        lumbar = center
        thorax = center + FNS.vert_up(truk_rot_horz, truk_rot_vert, 30)
        cervic = thorax + FNS.vert_up(truk_rot_horz, truk_rot_vert, 10)
        base = cervic + FNS.vert_up(truk_rot_horz + neck_rot_horz, truk_rot_vert + neck_rot_vert, 5)

        left_hip = sacrum + FNS.latr_left(0, 0, 5)
        right_hip = sacrum + FNS.latr_right(0, 0, 5)

        (left_hip_rot_vert, left_hip_rot_horz), (left_knee_rot_vert, left_knee_rot_horz), \
        (left_ankle_rot_vert, left_ankle_rot_horz) = self.append_data[1][0]

        (right_hip_rot_vert, right_hip_rot_horz), (right_knee_rot_vert, right_knee_rot_horz), \
        (right_ankle_rot_vert, right_ankle_rot_horz) = self.append_data[1][1]

        left_knee = left_hip + FNS.vert_down(left_hip_rot_horz, left_hip_rot_vert, 20)

        left_ankle = left_knee + FNS.vert_down(left_hip_rot_horz, left_hip_rot_vert + left_knee_rot_vert, 15)

        left_foot = left_ankle + FNS.vert_down(left_hip_rot_horz, left_hip_rot_vert + left_knee_rot_vert +
                                               left_ankle_rot_vert + np.pi / 2, 5)

        right_knee = right_hip + FNS.vert_down(right_hip_rot_horz, right_hip_rot_vert, 20)

        right_ankle = right_knee + FNS.vert_down(right_hip_rot_horz, right_hip_rot_vert + right_knee_rot_vert, 15)

        right_foot = right_ankle + FNS.vert_down(right_hip_rot_horz, right_hip_rot_vert + right_knee_rot_vert +
                                                 right_ankle_rot_vert + np.pi / 2, 5)

        return left_foot, right_foot

    # compute external torque for force of gravity and ground reaction force from joint positions of lower limbs
    def ext_forc(self, shift):
        FNS = self.FNS
        dep = 0.2 * self.default
        fof = np.zeros((2, 3, 2))
        grf = np.zeros((2, 3, 2))

        base, cervic, thorax, lumbar, sacrum = np.transpose(self.column_cpt(shift)[0], (1, 0))

        left_hip, left_knee, left_ankle, left_foot = np.transpose(self.lowlimb_cpt(shift)[0], (1, 0))
        right_hip, right_knee, right_ankle, right_foot = np.transpose(self.lowlimb_cpt(shift)[1], (1, 0))

        # magnitude of external force
        mass = (50 + 5 + 20) * 0.001

        # moment arm of force of gravity
        CoM = np.array((lumbar[0], lumbar[1], left_hip[2]))
        moment = np.linalg.norm(left_hip - CoM)
        fof[0][0][0] = moment * mass

        CoM = np.array((lumbar[0], lumbar[1], left_knee[2]))
        moment = np.linalg.norm(left_knee - CoM)
        fof[0][1][0] = moment * mass

        CoM = np.array((lumbar[0], lumbar[1], left_ankle[2]))
        moment = np.linalg.norm(left_ankle - CoM)
        fof[0][2][0] = moment * mass

        CoM = np.array((lumbar[0], lumbar[1], right_hip[2]))
        moment = np.linalg.norm(right_hip - CoM)
        fof[1][0][0] = moment * mass

        CoM = np.array((lumbar[0], lumbar[1], right_knee[2]))
        moment = np.linalg.norm(right_knee - CoM)
        fof[1][1][0] = moment * mass

        CoM = np.array((lumbar[0], lumbar[1], right_ankle[2]))
        moment = np.linalg.norm(right_ankle - CoM)
        fof[1][2][0] = moment * mass

        self.MapVar.fof = fof

        # moment arm of ground reaction force
        left_cond = FNS.delta_fn(FNS.cond_fn(left_ankle[2], -dep), 1)
        right_cond = FNS.delta_fn(FNS.cond_fn(right_ankle[2], -dep), 1)

        # both feet on ground
        if left_cond == 1 and right_cond == 1:
            mid_dist = np.linalg.norm(left_ankle - right_ankle) / 2
            cent = left_ankle + 0.5 * (right_ankle - left_ankle)

            CoP = np.array((cent[0], cent[1], left_ankle[2]))
            moment = np.linalg.norm(left_ankle - CoP)
            grf[0][2][0] = moment * mass

            CoP = np.array((cent[0], cent[1], left_knee[2]))
            moment = np.linalg.norm(left_knee - CoP)
            grf[0][1][0] = moment * mass

            CoP = np.array((cent[0], cent[1], left_hip[2]))
            moment = np.linalg.norm(left_hip - CoP)
            grf[0][0][0] = moment * mass

            CoP = np.array((cent[0], cent[1], right_ankle[2]))
            moment = np.linalg.norm(right_ankle - CoP)
            grf[1][2][0] = moment * mass

            CoP = np.array((cent[0], cent[1], right_knee[2]))
            moment = np.linalg.norm(right_knee - CoP)
            grf[1][1][0] = moment * mass

            CoP = np.array((cent[0], cent[1], right_hip[2]))
            moment = np.linalg.norm(right_hip - CoP)
            grf[1][0][0] = moment * mass

        # only left foot on ground
        if left_cond == 1 and right_cond != 1:
            CoP = np.array((left_ankle[0], left_ankle[1], left_ankle[2]))
            moment = np.linalg.norm(left_ankle - CoP)
            grf[0][2][0] = moment * mass

            CoP = np.array((left_ankle[0], left_ankle[1], left_knee[2]))
            moment = np.linalg.norm(left_knee - CoP)
            grf[0][1][0] = moment * mass

            CoP = np.array((left_ankle[0], left_ankle[1], left_hip[2]))
            moment = np.linalg.norm(left_hip - CoP)
            grf[0][0][0] = moment * mass

        # only right foot on ground
        if left_cond != 1 and right_cond == 1:
            CoP = np.array((right_ankle[0], right_ankle[1], right_ankle[2]))
            moment = np.linalg.norm(right_ankle - CoP)
            grf[1][2][0] = moment * mass

            CoP = np.array((right_ankle[0], right_ankle[1], right_knee[2]))
            moment = np.linalg.norm(right_knee - CoP)
            grf[1][1][0] = moment * mass

            CoP = np.array((right_ankle[0], right_ankle[1], right_hip[2]))
            moment = np.linalg.norm(right_hip - CoP)
            grf[1][0][0] = moment * mass

        self.MapVar.grf = grf

    # compute muscle origins and insertions of column muscles
    def col_cpt(self, shift):
        FNS = self.FNS

        if shift == 0:
            neck_rot_vert, neck_rot_horz = self.neck_rot
            truk_rot_vert, truk_rot_horz = self.trunk_rot

        else:
            self.neck_rot, self.trunk_rot = self.axial_data
            neck_rot_vert, neck_rot_horz = self.neck_rot
            truk_rot_vert, truk_rot_horz = self.trunk_rot

        base, cervic, thorax, lumbar, sacrum = np.transpose(self.column_cpt(shift)[0], (1, 0))

        # compute muscle origins of neck muscles
        base_plane = FNS.transv_plane(base, truk_rot_horz + neck_rot_horz, truk_rot_vert + neck_rot_vert, 2)

        base_left = base + FNS.latr_left(truk_rot_horz + neck_rot_horz, 0, 2)
        base_right = base + FNS.latr_right(truk_rot_horz + neck_rot_horz, 0, 2)
        base_front = base + FNS.latr_front(truk_rot_horz + neck_rot_horz, truk_rot_vert + neck_rot_vert, 2)
        base_back = base + FNS.latr_back(truk_rot_horz + neck_rot_horz, -(truk_rot_vert + neck_rot_vert), 2)

        left_back, right_back, left_front, right_front = base_plane

        base_frame = np.transpose(np.array((left_front, base_front, right_front, base_front,
                                            base, base_left, base, base_right, base,
                                            base_back, left_back, base_back, right_back)), (1, 0))


        # compute muscle insertions of neck muscles
        thorax_plane = FNS.transv_plane(thorax, truk_rot_horz, truk_rot_vert, 2)

        thorax_left = thorax + FNS.latr_left(truk_rot_horz, 0, 2)
        thorax_right = thorax + FNS.latr_right(truk_rot_horz, 0, 2)
        thorax_front = thorax + FNS.latr_front(truk_rot_horz, truk_rot_vert, 2)
        thorax_back = thorax + FNS.latr_back(truk_rot_horz, -(truk_rot_vert), 2)

        left_back, right_back, left_front, right_front = thorax_plane

        thorax_frame = np.transpose(np.array((left_front, thorax_front, right_front, thorax_front,
                                              thorax, thorax_left, thorax, thorax_right, thorax,
                                              thorax_back, left_back, thorax_back, right_back)), (1, 0))

        # compute muscle origins and insertions of neck muscles
        left_back_top, right_back_top, left_front_top, right_front_top = base_plane
        left_back_bot, right_back_bot, left_front_bot, right_front_bot = thorax_plane

        left_neck_mus = np.transpose(
            np.array((left_front_bot, left_back_top, left_front_top, left_back_bot, left_front_bot)), (1, 0))
        right_neck_mus = np.transpose(
            np.array((right_front_bot, right_back_top, right_front_top, right_back_bot, right_front_bot)), (1, 0))

        # compute muscle lengths of neck muscles
        neck_aR = right_back_top - right_front_bot
        neck_aL = left_back_top - left_front_bot
        neck_bR = right_front_top - right_back_bot
        neck_bL = left_front_top - left_back_bot

        aL_len = FNS.mus_len(neck_aL, 'single')
        aR_len = FNS.mus_len(neck_aR, 'single')
        bL_len = FNS.mus_len(neck_bL, 'single')
        bR_len = FNS.mus_len(neck_bR, 'single')

        #------------------------------------------------------------------------------------------------------------

        left_rot_vert, left_rot_horz = self.lowlimb_rot[0][0]
        right_rot_vert, right_rot_horz = self.lowlimb_rot[1][0]

        left_hip = sacrum + FNS.latr_left(0, 0, 5)
        right_hip = sacrum + FNS.latr_right(0, 0, 5)

        left_knee = left_hip + FNS.vert_down(left_rot_horz, left_rot_vert, 20)
        right_knee = right_hip + FNS.vert_down(right_rot_horz, right_rot_vert, 20)

        # compute muscle origins of pelvic muscles
        lumbar_plane = FNS.transv_plane(lumbar, truk_rot_horz, truk_rot_vert, 2)

        lumbar_left = lumbar + FNS.latr_left(truk_rot_horz, 0, 2)
        lumbar_right = lumbar + FNS.latr_right(truk_rot_horz, 0, 2)
        lumbar_front = lumbar + FNS.latr_front(truk_rot_horz, truk_rot_vert, 2)
        lumbar_back = lumbar + FNS.latr_back(truk_rot_horz, -truk_rot_vert, 2)

        left_back, right_back, left_front, right_front = lumbar_plane

        lumbar_frame = np.transpose(np.array((left_front, lumbar_front, right_front, lumbar_front,
                                              lumbar, lumbar_left, lumbar, lumbar_right, lumbar,
                                              lumbar_back, left_back, lumbar_back, right_back)), (1, 0))

        # compute muscle insertions of pelvic muscles
        left_insert_back = sacrum + FNS.latr_left(0, 0, 2)
        right_insert_back = sacrum + FNS.latr_right(0, 0, 2)
        left_insert_front = left_hip + FNS.vert_down(left_rot_horz, left_rot_vert, 2)
        right_insert_front = right_hip + FNS.vert_down(right_rot_horz, right_rot_vert, 2)

        # compute muscle origins and insertions of pelvic muscles
        left_trunk_mus = np.transpose(np.array((left_insert_front, left_back, left_front, left_insert_back)), (1, 0))
        right_trunk_mus = np.transpose(np.array((right_insert_front, right_back, right_front, right_insert_back)), (1, 0))

        # compute muscle lengths of pelvic muscles
        truk_aR = right_back - right_insert_front
        truk_aL = left_back - left_insert_front
        truk_bR = right_front - right_insert_back
        truk_bL = left_front - left_insert_back

        cL_len = FNS.mus_len(truk_aL, 'single')
        cR_len = FNS.mus_len(truk_aR, 'single')
        dL_len = FNS.mus_len(truk_bL, 'single')
        dR_len = FNS.mus_len(truk_bR, 'single')

        # ------------------------------------------------------------------------------------------------------------

        column = np.transpose(np.array((base, cervic, thorax, lumbar, sacrum)), (1, 0))
        left_limb = np.transpose(np.array((sacrum, left_hip, left_knee)), (1, 0))
        right_limb = np.transpose(np.array((sacrum, right_hip, right_knee)), (1, 0))
        joints = np.transpose(np.array((cervic, sacrum, left_hip, right_hip)), (1, 0))
        inserts = np.transpose(
            np.array((*base_plane, left_insert_back, right_insert_back, left_insert_front, right_insert_front)), (1, 0))

        return (base_frame, thorax_frame, lumbar_frame), (left_neck_mus, right_neck_mus), (left_trunk_mus, right_trunk_mus), \
               (column, left_limb, right_limb), (joints, inserts), \
               (((neck_aL, neck_aR), (neck_bR, neck_bL)), ((truk_aL, truk_aR), (truk_bR, truk_bL)))


    # draw column muscles from muscle origins to insertions
    def col_plt(self, base_frame, thorax_frame, lumbar_frame, left_neck_mus, right_neck_mus, left_trunk_mus,
                right_trunk_mus, column, left_limb, right_limb, joints, inserts):

        self.MapVar.base_frame.set_data(base_frame[0], base_frame[1])
        self.MapVar.base_frame.set_3d_properties(base_frame[2])

        self.MapVar.thorax_frame.set_data(thorax_frame[0], thorax_frame[1])
        self.MapVar.thorax_frame.set_3d_properties(thorax_frame[2])

        self.MapVar.lumbar_frame.set_data(lumbar_frame[0], lumbar_frame[1])
        self.MapVar.lumbar_frame.set_3d_properties(lumbar_frame[2])

        self.MapVar.left_neck_mus.set_data(left_neck_mus[0], left_neck_mus[1])
        self.MapVar.left_neck_mus.set_3d_properties(left_neck_mus[2])

        self.MapVar.right_neck_mus.set_data(right_neck_mus[0], right_neck_mus[1])
        self.MapVar.right_neck_mus.set_3d_properties(right_neck_mus[2])

        self.MapVar.left_trunk_mus.set_data(left_trunk_mus[0], left_trunk_mus[1])
        self.MapVar.left_trunk_mus.set_3d_properties(left_trunk_mus[2])

        self.MapVar.right_trunk_mus.set_data(right_trunk_mus[0], right_trunk_mus[1])
        self.MapVar.right_trunk_mus.set_3d_properties(right_trunk_mus[2])

        self.MapVar.column.set_data(column[0], column[1])
        self.MapVar.column.set_3d_properties(column[2])

        self.MapVar.left_limb.set_data(left_limb[0], left_limb[1])
        self.MapVar.left_limb.set_3d_properties(left_limb[2])

        self.MapVar.right_limb.set_data(right_limb[0], right_limb[1])
        self.MapVar.right_limb.set_3d_properties(right_limb[2])

        self.MapVar.left_joints.set_data(joints[0], joints[1])
        self.MapVar.left_joints.set_3d_properties(joints[2])

        self.MapVar.left_inserts.set_data(inserts[0], inserts[1])
        self.MapVar.left_inserts.set_3d_properties(inserts[2])

    # compute muscle origins and insertions of left arm muscles
    def left_arm_cpt(self, shift):
        FNS = self.FNS
        zeros = np.zeros(3)

        if shift == 0:
            (left_shoul_rot_vert, left_shoul_rot_horz), (left_elbow_rot_vert, left_elbow_rot_horz), \
            (left_wrist_rot_vert, left_wrist_rot_horz) = self.uplimb_rot[0]

        else:
            self.uplimb_rot = self.append_data[0]

            (left_shoul_rot_vert, left_shoul_rot_horz), (left_elbow_rot_vert, left_elbow_rot_horz), \
            (left_wrist_rot_vert, left_wrist_rot_horz) = self.uplimb_rot[0]

        base, cervic, thorax, lumbar, sacrum = np.transpose(self.column_cpt(shift)[0], (1, 0))

        left_shoul, left_elbow, left_wrist, left_hand = np.transpose(self.uplimb_cpt(shift)[0], (1, 0))

        # compute muscle origins of shoulder muscles
        pect_cent = thorax + FNS.latr_left(0, 0, 5)
        pect_plane = FNS.left_plane(pect_cent, 0, 0, 2)

        pect_left = pect_cent + FNS.latr_front(0, 0, 2)
        pect_right = pect_cent + FNS.latr_back(0, 0, 2)
        pect_front = pect_cent + FNS.latr_right(0, 0, 2)
        pect_back = pect_cent + FNS.latr_left(0, 0, 2)

        left_back, right_back, left_front, right_front = pect_plane

        pect_frame = np.transpose(np.array((left_front, pect_front, right_front, pect_front,
                                            pect_cent, pect_left, pect_cent, pect_right, pect_cent,
                                            pect_back, left_back, pect_back, right_back)), (1, 0))


        # compute muscle insertions of shoulder muscles
        humr_cent = left_shoul + FNS.vert_down(left_shoul_rot_horz, left_shoul_rot_vert, 5)
        humr_plane = FNS.front_plane(humr_cent, left_shoul_rot_horz, left_shoul_rot_vert, 0)

        humr_left = humr_cent + FNS.latr_left(left_shoul_rot_horz, 0, 0)
        humr_right = humr_cent + FNS.latr_right(left_shoul_rot_horz, 0, 0)
        humr_up = humr_cent + FNS.vert_up(left_shoul_rot_horz, left_shoul_rot_vert, 2)
        humr_down = humr_cent + FNS.vert_down(left_shoul_rot_horz, left_shoul_rot_vert, 2)

        left_down, right_down, left_up, right_up = humr_plane

        humr_frame = np.transpose(np.array((left_up, humr_up, right_up, humr_up,
                                            humr_cent, humr_left, humr_cent, humr_right, humr_cent,
                                            humr_down, left_down, humr_down, right_down)), (1, 0))

        # compute muscle origins and insertions of shoulder muscles
        right_up = humr_up
        left_up = humr_up
        right_down = humr_down
        left_down = humr_down

        shoul_horz = np.transpose(np.array((left_back, right_up, left_up, right_back)), (1, 0))
        shoul_vert = np.transpose(np.array((left_front, right_down, left_down, right_front)), (1, 0))

        # compute muscle lengths of shoulder muscles
        shoul_bR = right_up - left_back
        shoul_bL = left_up - right_back
        shoul_aR = right_down - left_front
        shoul_aL = left_down - right_front

        aR_len = FNS.mus_len(shoul_aR, 'single')
        aL_len = FNS.mus_len(shoul_aL, 'single')
        bR_len = FNS.mus_len(shoul_bR, 'single')
        bL_len = FNS.mus_len(shoul_bL, 'single')


        #------------------------------------------------------------------------------------------------------------
        # compute muscle origins of elbow muscles
        shoul_line = FNS.sagit_line(left_shoul, 0, 0, 2)
        shoul_front, shoul_back = shoul_line
        shoul_frame = np.transpose(np.array((shoul_front, left_shoul, shoul_back)), (1, 0))

        # compute muscle insertions of elbow muscles
        elbow_insert = left_elbow + FNS.vert_down(left_shoul_rot_horz, left_shoul_rot_vert + left_elbow_rot_vert, 2)

        # compute muscle origins and insertions of elbow muscles
        elbow_vert = np.transpose(np.array((shoul_line[0], elbow_insert, shoul_line[1])), (1, 0))

        # compute muscle lengths of elbow muscles
        elbow_aR = shoul_front - elbow_insert
        elbow_aL = shoul_back - elbow_insert

        cR_len = FNS.mus_len(elbow_aR, 'single')
        cL_len = FNS.mus_len(elbow_aL, 'single')

        # ------------------------------------------------------------------------------------------------------------
        # compute muscle origins of wrist muscles
        elbow_line = FNS.sagit_line(left_elbow + FNS.vert_up(left_shoul_rot_horz, left_shoul_rot_vert, 2),
                                    left_shoul_rot_horz, left_shoul_rot_vert, 2)
        elbow_front, elbow_back = elbow_line
        elbow_frame = np.transpose(np.array((elbow_front,
                                             left_elbow + FNS.vert_up(left_shoul_rot_horz, left_shoul_rot_vert, 2),
                                             elbow_back)), (1, 0))

        # compute muscle insertions of wrist muscles
        wrist_insert = left_wrist + FNS.vert_down(left_shoul_rot_horz,
                                                  left_shoul_rot_vert + left_elbow_rot_vert + left_wrist_rot_vert, 1)

        # compute muscle origins and insertions of wrist muscles
        wrist_vert = np.transpose(np.array((elbow_line[0], wrist_insert, elbow_line[1])), (1, 0))

        # compute muscle lengths of wrist muscles
        wrist_aR = elbow_front - wrist_insert
        wrist_aL = elbow_back - wrist_insert

        dR_len = FNS.mus_len(wrist_aR, 'single')
        dL_len = FNS.mus_len(wrist_aL, 'single')

        # ------------------------------------------------------------------------------------------------------------

        left_limb = np.transpose(np.array((thorax, left_shoul, left_elbow, left_wrist, left_hand)), (1, 0))
        left_inserts = np.transpose(np.array((left_up, right_up, left_down, right_down, elbow_insert, wrist_insert)),
                                    (1, 0))
        left_joints = np.transpose(np.array((left_shoul, left_elbow, left_wrist)), (1, 0))

        return (pect_frame, humr_frame), (shoul_frame, elbow_frame), (left_limb, left_joints, left_inserts), \
               (shoul_horz, shoul_vert), (elbow_vert, wrist_vert), \
               (((shoul_aR, shoul_aL), (shoul_bR, shoul_bL)), ((elbow_aR, elbow_aL), (zeros, zeros)),
                ((wrist_aR, wrist_aL), (zeros, zeros)))


    # draw left arm muscles from muscle origins to insertions
    def left_arm_plt(self, pect_frame, humr_frame, shoul_frame, elbow_frame, left_limb, left_joints, left_inserts,
                     shoul_horz, shoul_vert, elbow_vert, wrist_vert):
        self.MapVar.humr_frame.set_data(humr_frame[0], humr_frame[1])
        self.MapVar.humr_frame.set_3d_properties(humr_frame[2])

        self.MapVar.pect_frame.set_data(pect_frame[0], pect_frame[1])
        self.MapVar.pect_frame.set_3d_properties(pect_frame[2])

        self.MapVar.shoul_frame.set_data(shoul_frame[0], shoul_frame[1])
        self.MapVar.shoul_frame.set_3d_properties(shoul_frame[2])

        self.MapVar.elbow_frame.set_data(elbow_frame[0], elbow_frame[1])
        self.MapVar.elbow_frame.set_3d_properties(elbow_frame[2])

        self.MapVar.left_limb.set_data(left_limb[0], left_limb[1])
        self.MapVar.left_limb.set_3d_properties(left_limb[2])

        self.MapVar.left_inserts.set_data(left_inserts[0], left_inserts[1])
        self.MapVar.left_inserts.set_3d_properties(left_inserts[2])

        self.MapVar.left_joints.set_data(left_joints[0], left_joints[1])
        self.MapVar.left_joints.set_3d_properties(left_joints[2])

        self.MapVar.elbow_vert.set_data(elbow_vert[0], elbow_vert[1])
        self.MapVar.elbow_vert.set_3d_properties(elbow_vert[2])

        self.MapVar.wrist_vert.set_data(wrist_vert[0], wrist_vert[1])
        self.MapVar.wrist_vert.set_3d_properties(wrist_vert[2])

        self.MapVar.shoul_horz.set_data(shoul_horz[0], shoul_horz[1])
        self.MapVar.shoul_horz.set_3d_properties(shoul_horz[2])

        self.MapVar.shoul_vert.set_data(shoul_vert[0], shoul_vert[1])
        self.MapVar.shoul_vert.set_3d_properties(shoul_vert[2])


    # compute muscle origins and insertions of right arm muscles
    def right_arm_cpt(self, shift):
        FNS = self.FNS
        zeros = np.zeros(3)

        if shift == 0:
            (right_shoul_rot_vert, right_shoul_rot_horz), (right_elbow_rot_vert, right_elbow_rot_horz), \
            (right_wrist_rot_vert, right_wrist_rot_horz) = self.uplimb_rot[1]

        else:
            self.uplimb_rot = self.append_data[0]

            (right_shoul_rot_vert, right_shoul_rot_horz), (right_elbow_rot_vert, right_elbow_rot_horz), \
            (right_wrist_rot_vert, right_wrist_rot_horz) = self.uplimb_rot[1]

        base, cervic, thorax, lumbar, sacrum = np.transpose(self.column_cpt(shift)[0], (1, 0))

        right_shoul, right_elbow, right_wrist, right_hand = np.transpose(self.uplimb_cpt(shift)[1], (1, 0))

        # compute muscle origins of shoulder muscles
        pect_cent = thorax + FNS.latr_right(0, 0, 5)
        pect_plane = FNS.right_plane(pect_cent, 0, 0, 2)

        pect_left = pect_cent + FNS.latr_back(0, 0, 2)
        pect_right = pect_cent + FNS.latr_front(0, 0, 2)
        pect_front = pect_cent + FNS.latr_left(0, 0, 2)
        pect_back = pect_cent + FNS.latr_right(0, 0, 2)

        left_back, right_back, left_front, right_front = pect_plane

        pect_frame = np.transpose(np.array((left_front, pect_front, right_front, pect_front,
                                            pect_cent, pect_left, pect_cent, pect_right, pect_cent,
                                            pect_back, left_back, pect_back, right_back)), (1, 0))

        # compute muscle insertions of shoulder muscles
        humr_cent = right_shoul + FNS.vert_down(right_shoul_rot_horz, right_shoul_rot_vert, 5)
        humr_plane = FNS.front_plane(humr_cent, right_shoul_rot_horz, right_shoul_rot_vert, 0)

        humr_left = humr_cent + FNS.latr_left(right_shoul_rot_horz, 0, 0)
        humr_right = humr_cent + FNS.latr_right(right_shoul_rot_horz, 0, 0)
        humr_up = humr_cent + FNS.vert_up(right_shoul_rot_horz, right_shoul_rot_vert, 2)
        humr_down = humr_cent + FNS.vert_down(right_shoul_rot_horz, right_shoul_rot_vert, 2)


        left_down, right_down, left_up, right_up = humr_plane

        humr_frame = np.transpose(np.array((left_up, humr_up, right_up, humr_up,
                                            humr_cent, humr_left, humr_cent, humr_right, humr_cent,
                                            humr_down, left_down, humr_down, right_down)), (1, 0))

        # compute muscle origins and insertions of shoulder muscles
        right_up = humr_up
        left_up = humr_up
        right_down = humr_down
        left_down = humr_down

        shoul_horz = np.transpose(np.array((right_back, left_up, right_up, left_back)), (1, 0))
        shoul_vert = np.transpose(np.array((right_front, left_down, right_down, left_front)), (1, 0))

        # compute muscle lengths of shoulder muscles
        shoul_bR = left_up - right_back
        shoul_bL = right_up - left_back
        shoul_aR = left_down - right_front
        shoul_aL = right_down - left_front

        aR_len = FNS.mus_len(shoul_aR, 'single')
        aL_len = FNS.mus_len(shoul_aL, 'single')
        bL_len = FNS.mus_len(shoul_bL, 'single')
        bR_len = FNS.mus_len(shoul_bR, 'single')

        # ------------------------------------------------------------------------------------------------------------
        # compute muscle origins of elbow muscles
        shoul_line = FNS.sagit_line(right_shoul, 0, 0, 2)
        shoul_front, shoul_back = shoul_line
        shoul_frame = np.transpose(np.array((shoul_front, right_shoul, shoul_back)), (1, 0))

        # compute muscle insertions of elbow muscles
        elbow_insert = right_elbow + FNS.vert_down(right_shoul_rot_horz, right_shoul_rot_vert + right_elbow_rot_vert, 2)

        # compute muscle origins and insertions of elbow muscles
        elbow_vert = np.transpose(np.array((shoul_line[0], elbow_insert, shoul_line[1])), (1, 0))

        # compute muscle lengths of elbow muscles
        elbow_aR = shoul_front - elbow_insert
        elbow_aL = shoul_back - elbow_insert

        cR_len = FNS.mus_len(elbow_aR, 'single')
        cL_len = FNS.mus_len(elbow_aL, 'single')

        # ------------------------------------------------------------------------------------------------------------
        # compute muscle origins of wrist muscles
        elbow_line = FNS.sagit_line(right_elbow + FNS.vert_up(right_shoul_rot_horz, right_shoul_rot_vert, 2),
                                    right_shoul_rot_horz, right_shoul_rot_vert, 2)
        elbow_front, elbow_back = elbow_line
        elbow_frame = np.transpose(np.array((elbow_front,
                                             right_elbow + FNS.vert_up(right_shoul_rot_horz, right_shoul_rot_vert, 2),
                                             elbow_back)), (1, 0))

        # compute muscle insertions of wrist muscles
        wrist_insert = right_wrist + FNS.vert_down(right_shoul_rot_horz,
                                                   right_shoul_rot_vert + right_elbow_rot_vert + right_wrist_rot_vert, 1)

        # compute muscle origins and insertions of wrist muscles
        wrist_vert = np.transpose(np.array((elbow_line[0], wrist_insert, elbow_line[1])), (1, 0))

        # compute muscle lengths of wrist muscles
        wrist_aR = elbow_front - wrist_insert
        wrist_aL = elbow_back - wrist_insert

        dR_len = FNS.mus_len(wrist_aR, 'single')
        dL_len = FNS.mus_len(wrist_aL, 'single')

        # ------------------------------------------------------------------------------------------------------------

        right_limb = np.transpose(np.array((thorax, right_shoul, right_elbow, right_wrist, right_hand)), (1, 0))
        right_inserts = np.transpose(np.array((left_up, right_up, left_down, right_down, elbow_insert, wrist_insert)), (1, 0))
        right_joints = np.transpose(np.array((right_shoul, right_elbow, right_wrist)), (1, 0))

        return (pect_frame, humr_frame), (shoul_frame, elbow_frame), (right_limb, right_joints, right_inserts), \
               (shoul_horz, shoul_vert), (elbow_vert, wrist_vert), \
               (((shoul_aR, shoul_aL), (shoul_bL, shoul_bR)), ((elbow_aR, elbow_aL), (zeros, zeros)),
                ((wrist_aR, wrist_aL), (zeros, zeros)))

    # draw right arm muscles from muscle origins to insertions
    def right_arm_plt(self, pect_frame, humr_frame, shoul_frame, elbow_frame, right_limb, right_joints,
                      right_inserts, shoul_horz, shoul_vert, elbow_vert, wrist_vert):
        self.MapVar.humr_frame.set_data(humr_frame[0], humr_frame[1])
        self.MapVar.humr_frame.set_3d_properties(humr_frame[2])

        self.MapVar.pect_frame.set_data(pect_frame[0], pect_frame[1])
        self.MapVar.pect_frame.set_3d_properties(pect_frame[2])

        self.MapVar.shoul_frame.set_data(shoul_frame[0], shoul_frame[1])
        self.MapVar.shoul_frame.set_3d_properties(shoul_frame[2])

        self.MapVar.elbow_frame.set_data(elbow_frame[0], elbow_frame[1])
        self.MapVar.elbow_frame.set_3d_properties(elbow_frame[2])

        self.MapVar.right_limb.set_data(right_limb[0], right_limb[1])
        self.MapVar.right_limb.set_3d_properties(right_limb[2])

        self.MapVar.right_inserts.set_data(right_inserts[0], right_inserts[1])
        self.MapVar.right_inserts.set_3d_properties(right_inserts[2])

        self.MapVar.right_joints.set_data(right_joints[0], right_joints[1])
        self.MapVar.right_joints.set_3d_properties(right_joints[2])

        self.MapVar.elbow_vert.set_data(elbow_vert[0], elbow_vert[1])
        self.MapVar.elbow_vert.set_3d_properties(elbow_vert[2])

        self.MapVar.wrist_vert.set_data(wrist_vert[0], wrist_vert[1])
        self.MapVar.wrist_vert.set_3d_properties(wrist_vert[2])

        self.MapVar.shoul_horz.set_data(shoul_horz[0], shoul_horz[1])
        self.MapVar.shoul_horz.set_3d_properties(shoul_horz[2])

        self.MapVar.shoul_vert.set_data(shoul_vert[0], shoul_vert[1])
        self.MapVar.shoul_vert.set_3d_properties(shoul_vert[2])

    # compute left leg muscles from muscle origins to insertions
    def left_leg_cpt(self, shift):
        FNS = self.FNS
        zeros = np.zeros(3)

        if shift == 0:
            (left_hip_rot_vert, left_hip_rot_horz), (left_knee_rot_vert, left_knee_rot_horz), \
            (left_ankle_rot_vert, left_ankle_rot_horz) = self.lowlimb_rot[0]

        else:
            self.lowlimb_rot = self.append_data[1]

            (left_hip_rot_vert, left_hip_rot_horz), (left_knee_rot_vert, left_knee_rot_horz), \
            (left_ankle_rot_vert, left_ankle_rot_horz) = self.lowlimb_rot[0]

        base, cervic, thorax, lumbar, sacrum = np.transpose(self.column_cpt(shift)[0], (1, 0))

        left_hip, left_knee, left_ankle, left_foot = np.transpose(self.lowlimb_cpt(shift)[0], (1, 0))

        # compute muscle origins of hip muscles
        pelv_cent = sacrum + FNS.latr_left(0, 0, 2)
        pelv_plane = FNS.left_plane(pelv_cent, 0, 0, 2)

        pelv_left = pelv_cent + FNS.latr_front(0, 0, 2)
        pelv_right = pelv_cent + FNS.latr_back(0, 0, 2)
        pelv_front = pelv_cent + FNS.latr_right(0, 0, 2)
        pelv_back = pelv_cent + FNS.latr_left(0, 0, 2)

        left_back, right_back, left_front, right_front = pelv_plane

        pelv_frame = np.transpose(np.array((left_front, pelv_front, right_front, pelv_front,
                                            pelv_cent, pelv_left, pelv_cent, pelv_right, pelv_cent,
                                            pelv_back, left_back, pelv_back, right_back)), (1, 0))

        # compute muscle insertions of hip muscles
        femr_cent = left_hip + FNS.vert_down(left_hip_rot_horz, left_hip_rot_vert, 5)
        femr_plane = FNS.front_plane(femr_cent, left_hip_rot_horz, left_hip_rot_vert, 0)

        femr_left = femr_cent + FNS.latr_left(left_hip_rot_horz, 0, 0)
        femr_right = femr_cent + FNS.latr_right(left_hip_rot_horz, 0, 0)
        femr_up = femr_cent + FNS.vert_up(left_hip_rot_horz, left_hip_rot_vert, 2)
        femr_down = femr_cent + FNS.vert_down(left_hip_rot_horz, left_hip_rot_vert, 2)

        left_down, right_down, left_up, right_up = femr_plane

        femr_frame = np.transpose(np.array((left_up, femr_up, right_up, femr_up,
                                            femr_cent, femr_left, femr_cent, femr_right, femr_cent,
                                            femr_down, left_down, femr_down, right_down)), (1, 0))


        # compute muscle origins and insertions of hip muscles
        right_up = femr_up
        left_up = femr_up
        right_down = femr_down
        left_down = femr_down

        hip_horz = np.transpose(np.array((left_back, right_up, left_up, right_back)), (1, 0))
        hip_vert = np.transpose(np.array((left_front, right_down, left_down, right_front)), (1, 0))

        # compute muscle lengths of hip muscles
        hip_bR = right_up - left_back
        hip_bL = left_up - right_back
        hip_aR = right_down - left_front
        hip_aL = left_down - right_front

        aR_len = FNS.mus_len(hip_aR, 'single')
        aL_len = FNS.mus_len(hip_aL, 'single')
        bR_len = FNS.mus_len(hip_bR, 'single')
        bL_len = FNS.mus_len(hip_bL, 'single')

        # ------------------------------------------------------------------------------------------------------------
        # compute muscle origins of knee muscles
        hip_line = np.array((pelv_plane[2], pelv_plane[3]))
        hip_front, hip_back = hip_line
        hip_frame = np.transpose(np.array((hip_front, pelv_front, hip_back)), (1, 0))

        # compute muscle origins of knee muscles
        knee_insert = left_knee + FNS.vert_down(left_hip_rot_horz, left_hip_rot_vert + left_knee_rot_vert, 2)

        # compute muscle origins and insertions of knee muscles
        knee_vert = np.transpose(np.array((hip_line[0], knee_insert, hip_line[1])), (1, 0))

        # compute muscle lengths of knee muscles
        knee_aR = hip_front - knee_insert
        knee_aL = hip_back - knee_insert

        cR_len = FNS.mus_len(knee_aR, 'single')
        cL_len = FNS.mus_len(knee_aL, 'single')

        # ------------------------------------------------------------------------------------------------------------
        # compute muscle origins of ankle muscles
        knee_line = FNS.sagit_line(left_knee + FNS.vert_up(left_hip_rot_horz, left_hip_rot_vert, 2),
                                   left_hip_rot_horz, left_hip_rot_vert, 2)
        knee_front, knee_back = knee_line
        knee_frame = np.transpose(np.array((knee_front,
                                            left_knee + FNS.vert_up(left_hip_rot_horz, left_hip_rot_vert, 2), knee_back)), (1, 0))

        # compute muscle insertions of ankle muscles
        ankle_insert = left_ankle + FNS.vert_down(left_hip_rot_horz,
                                                  left_hip_rot_vert + left_knee_rot_vert + left_ankle_rot_vert + np.pi / 2, 1)

        # compute muscle origins and insertions of ankle muscles
        ankle_vert = np.transpose(np.array((knee_line[0], ankle_insert, knee_line[1])), (1, 0))

        # compute muscle lengths of ankle muscles
        ankle_aR = knee_front - ankle_insert
        ankle_aL = knee_back - ankle_insert

        dR_len = FNS.mus_len(ankle_aR, 'single')
        dL_len = FNS.mus_len(ankle_aL, 'single')

        # ------------------------------------------------------------------------------------------------------------

        left_limb = np.transpose(np.array((sacrum, left_hip, left_knee, left_ankle, left_foot)), (1, 0))
        left_inserts = np.transpose(np.array((left_up, right_up, left_down, right_down, knee_insert, ankle_insert)), (1, 0))
        left_joints = np.transpose(np.array((left_hip, left_knee, left_ankle)), (1, 0))

        return (pelv_frame, femr_frame), (hip_frame, knee_frame), (left_limb, left_joints, left_inserts), \
               (hip_horz, hip_vert), (knee_vert, ankle_vert), \
               (((hip_aR, hip_aL), (hip_bR, hip_bL)), ((knee_aR, knee_aL), (zeros, zeros)),
                ((ankle_aR, ankle_aL), (zeros, zeros)))

    # draw left leg muscles from muscle origins to insertions
    def left_leg_plt(self, pelv_frame, femr_frame, hip_frame, knee_frame, left_limb, left_joints, left_inserts,
                     hip_horz, hip_vert, knee_vert, ankle_vert):

        self.MapVar.femr_frame.set_data(femr_frame[0], femr_frame[1])
        self.MapVar.femr_frame.set_3d_properties(femr_frame[2])

        self.MapVar.pelv_frame.set_data(pelv_frame[0], pelv_frame[1])
        self.MapVar.pelv_frame.set_3d_properties(pelv_frame[2])

        self.MapVar.hip_frame.set_data(hip_frame[0], hip_frame[1])
        self.MapVar.hip_frame.set_3d_properties(hip_frame[2])

        self.MapVar.knee_frame.set_data(knee_frame[0], knee_frame[1])
        self.MapVar.knee_frame.set_3d_properties(knee_frame[2])

        self.MapVar.left_limb.set_data(left_limb[0], left_limb[1])
        self.MapVar.left_limb.set_3d_properties(left_limb[2])

        self.MapVar.left_inserts.set_data(left_inserts[0], left_inserts[1])
        self.MapVar.left_inserts.set_3d_properties(left_inserts[2])

        self.MapVar.left_joints.set_data(left_joints[0], left_joints[1])
        self.MapVar.left_joints.set_3d_properties(left_joints[2])

        self.MapVar.knee_vert.set_data(knee_vert[0], knee_vert[1])
        self.MapVar.knee_vert.set_3d_properties(knee_vert[2])

        self.MapVar.ankle_vert.set_data(ankle_vert[0], ankle_vert[1])
        self.MapVar.ankle_vert.set_3d_properties(ankle_vert[2])

        self.MapVar.hip_horz.set_data(hip_horz[0], hip_horz[1])
        self.MapVar.hip_horz.set_3d_properties(hip_horz[2])

        self.MapVar.hip_vert.set_data(hip_vert[0], hip_vert[1])
        self.MapVar.hip_vert.set_3d_properties(hip_vert[2])

    # compute muscle origins and insertions of right leg muscles
    def right_leg_cpt(self, shift):
        FNS = self.FNS
        zeros = np.zeros(3)

        if shift == 0:
            (right_hip_rot_vert, right_hip_rot_horz), (right_knee_rot_vert, right_knee_rot_horz), \
            (right_ankle_rot_vert, right_ankle_rot_horz) = self.lowlimb_rot[1]

        else:
            self.lowlimb_rot = self.append_data[1]

            (right_hip_rot_vert, right_hip_rot_horz), (right_knee_rot_vert, right_knee_rot_horz), \
            (right_ankle_rot_vert, right_ankle_rot_horz) = self.lowlimb_rot[1]

        base, cervic, thorax, lumbar, sacrum = np.transpose(self.column_cpt(shift)[0], (1, 0))

        right_hip, right_knee, right_ankle, right_foot = np.transpose(self.lowlimb_cpt(shift)[1], (1, 0))

        # compute muscle origins of hip muscles
        pelv_cent = sacrum + FNS.latr_right(0, 0, 2)
        pelv_plane = FNS.right_plane(pelv_cent, 0, 0, 2)

        pelv_left = pelv_cent + FNS.latr_back(0, 0, 2)
        pelv_right = pelv_cent + FNS.latr_front(0, 0, 2)
        pelv_front = pelv_cent + FNS.latr_left(0, 0, 2)
        pelv_back = pelv_cent + FNS.latr_right(0, 0, 2)

        left_back, right_back, left_front, right_front = pelv_plane

        pelv_frame = np.transpose(np.array((left_front, pelv_front, right_front, pelv_front,
                                            pelv_cent, pelv_left, pelv_cent, pelv_right, pelv_cent,
                                            pelv_back, left_back, pelv_back, right_back)), (1, 0))

        # compute muscle insertions of hip muscles
        femr_cent = right_hip + FNS.vert_down(right_hip_rot_horz, right_hip_rot_vert, 5)
        femr_plane = FNS.front_plane(femr_cent, right_hip_rot_horz, right_hip_rot_vert, 0)

        femr_left = femr_cent + FNS.latr_left(right_hip_rot_horz, 0, 0)
        femr_right = femr_cent + FNS.latr_right(right_hip_rot_horz, 0, 0)
        femr_up = femr_cent + FNS.vert_up(right_hip_rot_horz, right_hip_rot_vert, 2)
        femr_down = femr_cent + FNS.vert_down(right_hip_rot_horz, right_hip_rot_vert, 2)

        left_down, right_down, left_up, right_up = femr_plane

        femr_frame = np.transpose(np.array((left_up, femr_up, right_up, femr_up,
                                            femr_cent, femr_left, femr_cent, femr_right, femr_cent,
                                            femr_down, left_down, femr_down, right_down)), (1, 0))

        # compute muscle origins and insertions of hip muscles
        right_up = femr_up
        left_up = femr_up
        right_down = femr_down
        left_down = femr_down

        hip_horz = np.transpose(np.array((right_back, left_up, right_up, left_back)), (1, 0))
        hip_vert = np.transpose(np.array((right_front, left_down, right_down, left_front)), (1, 0))

        # compute muscle lengths of hip muscles
        hip_bR = left_up - right_back
        hip_bL = right_up - left_back
        hip_aR = left_down - right_front
        hip_aL = right_down - left_front

        aR_len = FNS.mus_len(hip_aR, 'single')
        aL_len = FNS.mus_len(hip_aL, 'single')
        bL_len = FNS.mus_len(hip_bL, 'single')
        bR_len = FNS.mus_len(hip_bR, 'single')

        # ------------------------------------------------------------------------------------------------------------
        # compute muscle origins of knee muscles
        hip_line = np.array((pelv_plane[2], pelv_plane[3]))
        hip_front, hip_back = hip_line
        hip_frame = np.transpose(np.array((hip_front, pelv_front, hip_back)), (1, 0))

        # compute muscle insertions of knee muscles
        knee_insert = right_knee + FNS.vert_down(right_hip_rot_horz, right_hip_rot_vert + right_knee_rot_vert, 2)

        # compute muscle origins and insertions of knee muscles
        knee_vert = np.transpose(np.array((hip_line[0], knee_insert, hip_line[1])), (1, 0))

        # compute muscle lengths of knee muscles
        knee_aR = hip_front - knee_insert
        knee_aL = hip_back - knee_insert

        cR_len = FNS.mus_len(knee_aR, 'single')
        cL_len = FNS.mus_len(knee_aL, 'single')

        # ------------------------------------------------------------------------------------------------------------
        # compute muscle origins of ankle muscles
        knee_line = FNS.sagit_line(right_knee + FNS.vert_up(right_hip_rot_horz, right_hip_rot_vert, 2),
                                   right_hip_rot_horz, right_hip_rot_vert, 2)
        knee_front, knee_back = knee_line
        knee_frame = np.transpose(np.array((knee_front,
                                            right_knee + FNS.vert_up(right_hip_rot_horz, right_hip_rot_vert, 2),
                                            knee_back)), (1, 0))

        # compute muscle insertions of ankle muscles
        ankle_insert = right_ankle + FNS.vert_down(right_hip_rot_horz,
                                                   right_hip_rot_vert + right_knee_rot_vert + right_ankle_rot_vert + np.pi / 2, 1)

        # compute muscle origins and insertions of ankle muscles
        ankle_vert = np.transpose(np.array((knee_line[0], ankle_insert, knee_line[1])), (1, 0))

        # compute muscle lengths of ankle muscles
        ankle_aR = knee_front - ankle_insert
        ankle_aL = knee_back - ankle_insert

        dR_len = FNS.mus_len(ankle_aR, 'single')
        dL_len = FNS.mus_len(ankle_aL, 'single')

        # ------------------------------------------------------------------------------------------------------------
        right_limb = np.transpose(np.array((sacrum, right_hip, right_knee, right_ankle, right_foot)), (1, 0))
        right_inserts = np.transpose(np.array((left_up, right_up, left_down, right_down, knee_insert, ankle_insert)), (1, 0))
        right_joints = np.transpose(np.array((right_hip, right_knee, right_ankle)), (1, 0))

        return (pelv_frame, femr_frame), (hip_frame, knee_frame), (right_limb, right_joints, right_inserts), \
               (hip_horz, hip_vert), (knee_vert, ankle_vert), \
               (((hip_aR, hip_aL), (hip_bL, hip_bR)), ((knee_aR, knee_aL), (zeros, zeros)),
                ((ankle_aR, ankle_aL), (zeros, zeros)))

    # draw right leg muscles from muscle origins to insertions
    def right_leg_plt(self, pelv_frame, femr_frame, hip_frame, knee_frame, right_limb, right_joints, right_inserts,
                      hip_horz, hip_vert, knee_vert, ankle_vert):

        self.MapVar.femr_frame.set_data(femr_frame[0], femr_frame[1])
        self.MapVar.femr_frame.set_3d_properties(femr_frame[2])

        self.MapVar.pelv_frame.set_data(pelv_frame[0], pelv_frame[1])
        self.MapVar.pelv_frame.set_3d_properties(pelv_frame[2])

        self.MapVar.hip_frame.set_data(hip_frame[0], hip_frame[1])
        self.MapVar.hip_frame.set_3d_properties(hip_frame[2])

        self.MapVar.knee_frame.set_data(knee_frame[0], knee_frame[1])
        self.MapVar.knee_frame.set_3d_properties(knee_frame[2])

        self.MapVar.right_limb.set_data(right_limb[0], right_limb[1])
        self.MapVar.right_limb.set_3d_properties(right_limb[2])

        self.MapVar.right_inserts.set_data(right_inserts[0], right_inserts[1])
        self.MapVar.right_inserts.set_3d_properties(right_inserts[2])

        self.MapVar.right_joints.set_data(right_joints[0], right_joints[1])
        self.MapVar.right_joints.set_3d_properties(right_joints[2])

        self.MapVar.knee_vert.set_data(knee_vert[0], knee_vert[1])
        self.MapVar.knee_vert.set_3d_properties(knee_vert[2])

        self.MapVar.ankle_vert.set_data(ankle_vert[0], ankle_vert[1])
        self.MapVar.ankle_vert.set_3d_properties(ankle_vert[2])

        self.MapVar.hip_horz.set_data(hip_horz[0], hip_horz[1])
        self.MapVar.hip_horz.set_3d_properties(hip_horz[2])

        self.MapVar.hip_vert.set_data(hip_vert[0], hip_vert[1])
        self.MapVar.hip_vert.set_3d_properties(hip_vert[2])







