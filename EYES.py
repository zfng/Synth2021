import numpy as np
from FUNCS import FNS, RK4
from PTSS import Ptss as ptss
from PTSS import PtssJoint as ptssjnt
from PTSS import PtssSpatial as ptssspt

# variable class for Eye Module
class EyeVar:
    def __init__(self, ret_size, num_deg):
        self.ret_size = ret_size
        self.num_deg = num_deg
        self.sc = self.Colliculus(ret_size)
        self.ret = self.Retina(ret_size)
        self.bs = self.Brainstem(ret_size)
        self.cbm = self.Cerebellum(ret_size)
        self.ppc = self.Parietal(ret_size, num_deg)
        self.snr = self.Substantia(ret_size)

    class Retina:
        def __init__(self, ret_size):
            self.ret_map = np.zeros((2, 2 * ret_size, 2 * ret_size))
            self.gaze_map = np.zeros((2, 2 * ret_size, 2 * ret_size))

    class Colliculus:
        def __init__(self, ret_size):
            self.top_layer = np.zeros((2, 2 * ret_size, 2 * ret_size))
            self.bot_layer = np.zeros((2, 2 * ret_size, 2 * ret_size))

    class Substantia():
        def __init__(self, ret_size):
            self.fixate_habit = np.ones((2))
            self.sc_top = np.ones((2, 2 * ret_size, 2 * ret_size))
            self.sc_bot = np.ones((2, 2 * ret_size, 2 * ret_size))

    class Cerebellum:
        def __init__(self, ret_size):
            self.ltm_sc_bs_ips = np.zeros((2, 2, 2, 2 * ret_size, 2 * ret_size))
            self.ltm_sc_bs_cot = np.zeros((2, 2, 2, 2 * ret_size, 2 * ret_size))
            self.sc_bs_ips = np.zeros((2, 2, 2, 2 * ret_size, 2 * ret_size))
            self.sc_bs_cot = np.zeros((2, 2, 2, 2 * ret_size, 2 * ret_size))
            self.cond_ips = np.zeros((2, 2, 2))
            self.cond_cot = np.zeros((2, 2, 2))
            self.uncond_ips = np.zeros((2, 2, 2))
            self.uncond_cot = np.zeros((2, 2, 2))
            self.cond_sc_bs = np.zeros((2, 2, 2))
            self.uncond_sc_bs = np.zeros((2, 2, 2))

    class Brainstem:
        def __init__(self, ret_size):
            self.long_burst = np.zeros((2, 2, 2))
            self.excit_burst = np.zeros((2, 2, 2))
            self.inhib_burst = np.zeros((2, 2, 2))
            self.tonic_eye = np.zeros((2, 2, 2))
            self.mus_eye = np.zeros((2, 2, 2))
            self.pauser_eye = np.ones((2))
            self.arousal_eye = np.ones((2))
            self.eyeball = ret_size * np.ones((2, 2), dtype=int)
            self.norm_eye = 1 / 2 * np.ones((2, 2, 2))

    class Parietal:
        def __init__(self, ret_size, num_deg):
            self.ltm_eye_sc = np.zeros((2, 2 * ret_size, 2 * ret_size, 2 * ret_size, 2 * ret_size))
            self.mtm_eye_sc = np.zeros((2, 2 * ret_size, 2 * ret_size, 2 * ret_size, 2 * ret_size))
            self.cond_eye_sc = np.zeros((2, 2 * ret_size, 2 * ret_size))

            self.ltm_targ_eye = np.zeros((num_deg, num_deg, num_deg, 2, 2, 2))
            self.dv_targ_eye = np.zeros((2, 2, 2))
            self.sv_targ_eye = np.zeros((2, 2, 2))
            self.targ_eye_est = np.zeros((2, 2, 2))

            self.ltm_bino_targ = np.zeros((3, 2, 2 * ret_size, 2 * ret_size, 2 * ret_size))
            self.dv_bino_targ = np.zeros((3, 2))
            self.spt_targ_eye = 0 / 2 * np.ones((3, 2))
            self.bino_eye_est = np.zeros((3, 2))

            self.ltm_targ_head = np.zeros((num_deg, num_deg, 3, 2))
            self.dv_targ_head = np.zeros((3, 2))
            self.spt_targ_head = 0 / 2 * np.ones((3, 2))
            self.bino_head_est = np.zeros((3, 2))

            self.spt_angle = 1 / 2 * np.ones((3, 2))
            self.spt_pres_eye = 1 / 2 * np.ones((3, 2))
            self.spt_targ = 0 / 2 * np.ones((3, 2))

            self.sptmp_eye = np.zeros((2, num_deg, num_deg))
            self.sptmp_head = np.zeros((num_deg, num_deg))
            self.sptmp_targ = np.zeros((num_deg, num_deg, num_deg))
            self.binoc_map = np.zeros((2 * ret_size, 2 * ret_size, 2 * ret_size))

            self.ppv_neck = 1 / 2 * np.ones((2, 2, 2))


# method class for Eye Module
class EyeFun:
    def __init__(self, EyeVar, Input):
        self.Eye = EyeVar
        self.Input = Input
        self.ste_size = 0.01
        self.FNS = FNS()
        self.RK4 = RK4()
        self.ret_size = EyeVar.ret_size
        self.num_deg = EyeVar.num_deg
        self.ptssspt = ptssspt(self.num_deg, self.num_deg, self.num_deg)
        self.ptss = ptss(self.num_deg, self.num_deg)
        self.ptssjnt = ptssjnt(self.num_deg, self.num_deg)

    # model Retina
    def Retina(self):
        size = self.Eye.ret_size
        FNS = self.FNS
        fixate_map = np.zeros((2, 2 * size, 2 * size))
        zeros = np.zeros((2, 2), dtype=int)

        # reset all non-dynamic maps
        self.Eye.ret.ret_map = np.zeros((2, 2 * size, 2 * size))
        self.Eye.ret.gaze_map = np.zeros((2, 2 * size, 2 * size))


        # activate retinas
        left_height, left_width = self.Input[0] + size
        right_height, right_width = self.Input[1] + size

        left_bound = FNS.retmap_bound(left_height, 2 * size), FNS.retmap_bound(left_width, 2 * size)
        right_bound = FNS.retmap_bound(right_height, 2 * size), FNS.retmap_bound(right_width, 2 * size)

        self.Eye.ret.ret_map[0][left_bound[0]][left_bound[1]] = 1
        self.Eye.ret.ret_map[1][right_bound[0]][right_bound[1]] = 1


        # compute gaze error
        input = np.array((left_bound, right_bound))
        ret_map = self.Eye.ret.ret_map
        fovea = np.array((self.Eye.bs.eyeball), dtype=np.int32)

        if np.array_equal(self.Input, zeros):  # no target
            self.Eye.ret.gaze_map = np.zeros((2, 2 * size, 2 * size))

        elif (FNS.compare(fovea[0], input[0]) == 1 and FNS.compare(fovea[1], input[1]) == 1):  # target near the foveae
            fixate_map[0][size][size] = 1
            fixate_map[1][size][size] = 1
            self.Eye.ret.gaze_map = fixate_map

        else:  # target not near the foveae
            for s in range(2):
                self.Eye.ret.gaze_map[s] = FNS.gaze_map_trans(fovea[s], ret_map[s], size)

    # model SuperiorColliculus
    def Colliculus(self, t, T1, T2, interval):
        size = self.Eye.ret_size
        step = self.ste_size
        FNS = self.FNS
        RK4 = self.RK4
        sc_top = FNS.thresh_fn(self.Eye.sc.top_layer, 0)
        sc_bot = FNS.thresh_fn(self.Eye.sc.bot_layer, 0)
        snr_top = FNS.thresh_fn(self.Eye.snr.sc_top, 0)
        snr_bot = FNS.thresh_fn(self.Eye.snr.sc_bot, 0)
        gaze = self.Eye.ret.gaze_map
        habit = self.Eye.snr.fixate_habit
        eye = self.Eye.ppc.cond_eye_sc

        # process gaze error in sc top and bottom layers
        for s in range(2):
            # sc top layer
            # excitatory inputs
            light_input = 10 * gaze[s]
            sc_bot_feedback = 3 * FNS.sigmoid_fn(sc_bot[s], 0.5, 2)

            # inhibitory inputs
            mrf_top_modulate = 1 * FNS.sample_fn(np.sum(sc_bot[s]) - sc_bot[s][size][size], 0)
            sc_bot_fixate = 5 * sc_bot[s][size][size]
            snr_top_modulate = 5 * FNS.sigmoid_fn(snr_top[s], 0.1, 2)

            self.Eye.sc.top_layer[s] = \
                RK4.rk4(sc_top[s], -1 * sc_top[s] +
                        (1 - sc_top[s]) * (light_input + sc_bot_feedback) -
                        (0.5 + sc_top[s]) * (mrf_top_modulate + sc_bot_fixate + snr_top_modulate), step)


            # sc bottom layer at non-fixation position
            # excitatory inputs
            eye_input = 1 * eye[s]
            sc_top_input = 10 * FNS.sc_map_cent(sc_top[s], 0.5, size)
            on_center = 3 * sc_bot[s]

            # inhibitory inputs
            mrf_bot_modulate = 1 * FNS.sample_fn(np.sum(sc_bot[s]) - sc_bot[s][size][size], 0)
            sc_bot_fixate = 1 * sc_bot[s][size][size]
            snr_bot_modulate = 5 * FNS.sigmoid_fn(snr_bot[s], 0.1, 2)
            off_surround = 2 * FNS.sc_map_surr(sc_bot[s], 2, size)

            self.Eye.sc.bot_layer[s] = \
                RK4.rk4(sc_bot[s], -0.1 * sc_bot[s] +
                        (1 - sc_bot[s]) * (eye_input + sc_top_input + on_center) -
                        sc_bot[s] * (mrf_bot_modulate + sc_bot_fixate + snr_bot_modulate + off_surround), step)


            # sc bottom layer at fixation position
            # excitatory inputs
            fixate_hold = 10 * FNS.forwd_period(t, T1, interval) + 10 * FNS.backw_period(t, T2, interval)
            fixate_modulate = 5 * habit[s]
            eye_input = 1 * eye[s][size][size]

            # inhibitory inputs
            off_surround = 5 * (np.sum(sc_bot[s] * FNS.gauss_kernel(size, size, 2, 2 * size)) - sc_bot[s][size][size])
            sc_top_inhib = 5 * (np.sum(sc_top[s]) - sc_top[s][size][size])

            self.Eye.sc.bot_layer[s][size][size] = \
                RK4.rk4(sc_bot[s][size][size], -0.1 * sc_bot[s][size][size] +
                        (1 - sc_bot[s][size][size]) * (fixate_hold + fixate_modulate + eye_input) -
                        sc_bot[s][size][size] * (off_surround + sc_top_inhib), step)

    # model BasalGanglia (SNr)
    def Substantia(self):
        size = self.ret_size
        RK4 = self.RK4
        step = self.ste_size
        gaze = self.Eye.ret.gaze_map
        habit = self.Eye.snr.fixate_habit
        top = self.Eye.snr.sc_top
        bot = self.Eye.snr.sc_bot

        # release tonic inhibition with light activation
        light = np.array([np.sum(gaze[s]) - gaze[s][size][size] for s in range(2)])
        self.Eye.snr.fixate_habit = RK4.rk4(habit, (1 - habit) - 10 * light * habit, step)

        for s in range(2):
            self.Eye.snr.sc_top[s] = \
                RK4.rk4(top[s], (1 - top[s]) * (1 + 5 * habit[s]) - (1 + top[s]) * (10 * gaze[s]), step)

            self.Eye.snr.sc_bot[s] = \
                RK4.rk4(bot[s], (1 - bot[s]) * (1 + 5 * habit[s]) - (1 + bot[s]) * (10 * light[s]), step)

    # model Cerebellum
    def Cerebellum(self):
        FNS = self.FNS
        RK4 = self.RK4
        ptss = self.ptss
        size = self.ret_size
        step = self.ste_size
        sc_top = FNS.thresh_fn(self.Eye.sc.top_layer, 0)
        sc_bot = FNS.thresh_fn(self.Eye.sc.bot_layer, 0)
        ltm_ips = self.Eye.cbm.ltm_sc_bs_ips
        ltm_cot = self.Eye.cbm.ltm_sc_bs_cot

        # reset non-dynamic maps and temporary variables
        self.Eye.cbm.sc_bs_ips = np.zeros((2, 2, 2, 2 * size, 2 * size))
        self.Eye.cbm.sc_bs_cot = np.zeros((2, 2, 2, 2 * size, 2 * size))
        self.Eye.cbm.cond_sc_bs = np.zeros((2, 2, 2))
        self.Eye.cbm.uncond_sc_bs = np.zeros((2, 2, 2))

        # ____________________________________________________________________________________________________________
        # convert gaze error to motor error
        mot_ips = FNS.argmax(sc_top[0], size)
        mot_cot = FNS.argmax(sc_top[1], size)
        gaz_ips = FNS.argmax(sc_bot[0], size)
        gaz_cot = FNS.argmax(sc_bot[1], size)
        mus_ips = ptss.ret_mus_tran(*mot_ips)
        mus_cot = ptss.ret_mus_tran(*mot_cot)
        mus = np.array((mus_ips, mus_cot))

        # ____________________________________________________________________________________________________________
        # sample motor error using sampling signal from sc_bot
        pt_max = gaz_ips + size, gaz_cot + size
        sc_max = FNS.arrmax(pt_max, size)

        for s in range(2):
            for m in range(2):
                for n in range(2):
                    self.Eye.cbm.ltm_sc_bs_ips[s][m][n] = \
                        RK4.rk4(ltm_ips[s][m][n], sc_max[s] * (-0.01 * ltm_ips[s][m][n] + mus[s][m][n]), step)

                    self.Eye.cbm.ltm_sc_bs_cot[s][m][n] = \
                        RK4.rk4(ltm_cot[s][m][n], sc_max[s] * (-0.01 * ltm_cot[s][m][n] + mus[(s + 1) % 2][m][n]), step)

                    self.Eye.cbm.sc_bs_ips[s][m][n] = sc_max[s] * mus[s][m][n]
                    self.Eye.cbm.sc_bs_cot[s][m][n] = sc_max[s] * mus[(s + 1) % 2][m][n]

        # ____________________________________________________________________________________________________________
        # compute conditioned signal
        ltm_ips = self.Eye.cbm.ltm_sc_bs_ips
        ltm_cot = self.Eye.cbm.ltm_sc_bs_cot

        diff_ltm_ips = FNS.thresh_fn(FNS.diff_mat(ltm_ips, 2 * size), 0)
        diff_ltm_cot = FNS.thresh_fn(FNS.diff_mat(ltm_cot, 2 * size), 0)

        self.Eye.cbm.cond_sc_bs = np.array([[[np.sum(sc_max[s] * diff_ltm_ips[s][m][n]) + \
                                              np.sum(sc_max[(s + 1) % 2] * diff_ltm_cot[(s + 1) % 2][m][n])
                                              for n in range(2)] for m in range(2)] for s in range(2)])


        # compute unconditioned signal
        pre_ips = self.Eye.cbm.sc_bs_ips
        pre_cot = self.Eye.cbm.sc_bs_cot

        diff_pre_ips = FNS.thresh_fn(FNS.diff_mat(pre_ips, 2 * size), 0)
        diff_pre_cot = FNS.thresh_fn(FNS.diff_mat(pre_cot, 2 * size), 0)

        self.Eye.cbm.uncond_sc_bs = np.array([[[np.sum(sc_max[s] * diff_pre_ips[s][m][n]) + \
                                                np.sum(sc_max[(s + 1) % 2] * diff_pre_cot[(s + 1) % 2][m][n])
                                                for n in range(2)] for m in range(2)] for s in range(2)])

    # model Brainstem
    def Brainstem(self):
        FNS = self.FNS
        RK4 = self.RK4
        size = self.ret_size
        step = self.ste_size
        sc_bot = FNS.thresh_fn(self.Eye.sc.bot_layer, 0)
        cond = FNS.thresh_fn(self.Eye.cbm.cond_sc_bs, 0)
        uncond = FNS.thresh_fn(self.Eye.cbm.uncond_sc_bs, 0)
        long_bur = FNS.thresh_fn(self.Eye.bs.long_burst, 0)
        rev_long = FNS.rev_mus(long_bur, 'eye')
        excit_bur = FNS.thresh_fn(self.Eye.bs.excit_burst, 0)
        rev_excit = FNS.rev_mus(excit_bur, 'eye')
        inhib_bur = FNS.thresh_fn(self.Eye.bs.inhib_burst, 0)
        pauser = FNS.thresh_fn(self.Eye.bs.pauser_eye, 0)
        tonic = FNS.thresh_fn(self.Eye.bs.tonic_eye, 0)
        rev_tonic = FNS.rev_mus(tonic, 'eye')
        arous = self.Eye.bs.arousal_eye
        mus = FNS.thresh_fn(self.Eye.bs.mus_eye, 0)
        norm = self.Eye.bs.norm_eye

        # process movement command that moves the eyes to the target
        for s in range(2):
            self.Eye.bs.long_burst[s] = RK4.rk4(long_bur[s], -1 * long_bur[s] +
                                                5 * cond[s] + 1 * uncond[s] - 3 * inhib_bur[s], step)

            # excitatory input to pauser
            fixate = 10 * sc_bot[s][size][size]

            # inhibitory input to pauser
            disinhib = FNS.sum_mus(FNS.sigmoid_fn(long_bur, 0.1, 2))[s]

            self.Eye.bs.pauser_eye[s] = RK4.rk4(pauser[s], -0.1 * pauser[s] +
                        (1 - pauser[s]) * (arous[s] + fixate) - 1 * (pauser[s] + 0.5) * disinhib, step)

            self.Eye.bs.excit_burst[s] = RK4.rk4(excit_bur[s], -2 * excit_bur[s] +
                        5 * long_bur[s] - 3 * rev_long[s] + arous[s] - 10 * FNS.sigmoid_fn(pauser[s], 0.1, 2), step)

            self.Eye.bs.inhib_burst[s] = RK4.rk4(inhib_bur[s], -1 * inhib_bur[s] + 2 * excit_bur[s], step)

            self.Eye.bs.tonic_eye[s] = RK4.rk4(tonic[s], -0.5 * tonic[s] +
                                               1 * excit_bur[s] - 0.5 * rev_excit[s], step)

            self.Eye.bs.mus_eye[s] = RK4.rk4(mus[s], -0.5 * mus[s] +
                                             1 * excit_bur[s] - 0.5 * rev_excit[s] + tonic[s], step)

            self.Eye.bs.norm_eye[s] = RK4.rk4(norm[s], -0.001 * norm[s] +
                                              (1 - norm[s]) * tonic[s] - norm[s] * rev_tonic[s], step)

        # parse position of foveae
        self.Eye.bs.eyeball = FNS.parse_eye(norm, size)

    # model ParietalCortex for connection to eye movement circuit
    def ParietalTarg(self):
        size = self.ret_size
        num = self.num_deg
        step = self.ste_size
        ptssspt = self.ptssspt
        ptts = self.ptss
        FNS = self.FNS
        RK4 = self.RK4
        parse_targ = FNS.parse_targ(self.Eye.ppc.spt_targ_eye, num, 'eye')
        ltm_err = self.Eye.ppc.ltm_targ_eye
        tran_ltm_err = np.transpose(ltm_err, (3, 4, 5, 0, 1, 2))
        dv_err = self.Eye.ppc.dv_targ_eye
        sv_err = self.Eye.ppc.sv_targ_eye

        ltm = self.Eye.ppc.ltm_eye_sc
        mtm = self.Eye.ppc.mtm_eye_sc
        tran_mtm = np.transpose(mtm, (0, 3, 4, 1, 2))
        sc_top = FNS.thresh_fn(self.Eye.sc.top_layer, 0)
        norm_eye = self.Eye.bs.norm_eye
        parse_dv = FNS.parse_eye(dv_err, size)

        # reset non-dynamic maps
        self.Eye.ppc.sptmp_eye = np.zeros((2, 2 * size, 2 * size))
        self.Eye.ppc.sptmp_targ = np.zeros((num, num, num))
        self.Eye.ppc.cond_eye_sc = np.zeros((2, 2 * size, 2 * size))

        # ____________________________________________________________________________________________________________
        # sample corresponding eye position
        a_max, b_max, r_max = parse_targ
        bound = ptssspt.ptssspt_bound(a_max, b_max, r_max, num, num, num)
        for a in bound[0]:
            for b in bound[1]:
                for r in bound[2]:
                    self.Eye.ppc.ltm_targ_eye[a][b][r] = \
                        RK4.rk4(ltm_err[a][b][r], 0 * ptssspt.ptssspt_gradient(a, b, r, a_max, b_max, r_max) *
                                (-0.1 * ltm_err[a][b][r] + norm_eye), step)

                    self.Eye.ppc.sptmp_targ[a][b][r] = ptssspt.ptssspt_gradient(a, b, r, a_max, b_max, r_max)

        sptmp_targ = self.Eye.ppc.sptmp_targ
        norm_est = np.array([[[np.sum(sptmp_targ * tran_ltm_err[s][m][n])
                               for n in range(2)] for m in range(2)] for s in range(2)])
        rev_est = FNS.rev_mus(norm_est, 'eye')
        self.Eye.ppc.sv_targ_eye = norm_est / (norm_est + rev_est + 0.01)

        # check learning of eye position
        self.Eye.ppc.targ_eye_est = norm_eye - sv_err


        # compute difference btw expected eye position and present eye position
        self.Eye.ppc.dv_targ_eye = norm_eye - sv_err

        # ____________________________________________________________________________________________________________
        # sample corresponding gaze error
        for s in range(2):
            b_max, a_max = parse_dv[s]
            bound = ptts.ptss_bound(b_max, a_max, 2 * size, 2 * size, '2')
            sc_top_input = 5 * FNS.sc_map_cent(sc_top[s], 0.5, size)

            for b in bound[0]:
                for a in bound[1]:
                    self.Eye.ppc.ltm_eye_sc[s][b][a] = \
                        RK4.rk4(ltm[s][b][a], 1.0 * ptts.ptss_gradient(b, a, b_max, a_max, '2') *
                                (-0.01 * ltm[s][b][a] + sc_top_input), step)

                    self.Eye.ppc.sptmp_eye[s][b][a] = ptts.ptss_gradient(b, a, b_max, a_max, '2')

                    sptmp_eye = self.Eye.ppc.sptmp_eye
                    self.Eye.ppc.mtm_eye_sc[s][b][a] = \
                        RK4.rk4(mtm[s][b][a], (ltm[s][b][a] - mtm[s][b][a]) - sptmp_eye[s][b][a] * mtm[s][b][a], step)

        sptmp_eye = self.Eye.ppc.sptmp_eye
        self.Eye.ppc.cond_eye_sc = np.array([[[np.sum(sptmp_eye[s] * tran_mtm[s][j][i])
                                               for i in range(2 * size)] for j in range(2 * size)] for s in range(2)])


    # model ParietalCortex for binocular fusion and spatial representation
    def ParietalSpat(self, t, interval):
        size = self.ret_size
        step = self.ste_size
        num = self.num_deg
        ptssjnt = self.ptssjnt
        FNS = self.FNS
        RK4 = self.RK4
        gaze = self.Eye.ret.gaze_map
        on = FNS.backw_period(t, 1, interval)

        eye_err = (self.Eye.bs.eyeball - size) - self.Input
        eye_hold = 1 - self.FNS.delta_fn(eye_err, 0)
        mus_eye = FNS.rev_row(self.Eye.bs.norm_eye)
        rev_mus = FNS.rev_mus(mus_eye, 'eye')
        ltm_bino_targ = self.Eye.ppc.ltm_bino_targ
        dv_bino_targ = self.Eye.ppc.dv_bino_targ
        spt_targ_eye = self.Eye.ppc.spt_targ_eye

        column = self.Eye.ppc.ppv_neck
        parse_neck = FNS.parse_axial(column, num)[0]
        head_orig = (num // 2) * np.ones(2)
        head_hold = 1 - self.FNS.delta_fn(head_orig, parse_neck)
        #head_hold = 1 - eye_hold
        ltm_targ_head = self.Eye.ppc.ltm_targ_head
        tran_ltm_targ_head = np.transpose(ltm_targ_head, (2, 3, 0, 1))
        dv_targ_head = self.Eye.ppc.dv_targ_head
        spt_targ_head = self.Eye.ppc.spt_targ_head

        # reset non-dynamic maps
        self.Eye.ppc.binoc_map = np.zeros((2 * size, 2 * size, 2 * size))
        self.Eye.ppc.sptmp_verg = np.zeros((num, num))
        self.Eye.ppc.sptmp_head = np.zeros((num, num))

        # ____________________________________________________________________________________________________________
        # compute binocular fusion
        binoc_fuse = FNS.binoc_fuse(gaze[0], gaze[1], size)
        self.Eye.ppc.binoc_map = self.FNS.binoc_map_trans(binoc_fuse, size)
        binoc_map = self.Eye.ppc.binoc_map

        # ____________________________________________________________________________________________________________
        # compute spatial representation of foveated target
        for k in range(2):
            self.Eye.ppc.spt_angle[k] = (mus_eye[0][k] + mus_eye[1][k]) / 2

        self.Eye.ppc.spt_angle[2] = rev_mus[1][0] - rev_mus[0][0]
        self.Eye.ppc.spt_angle[2][0] = FNS.thresh_fn(0.5 + self.Eye.ppc.spt_angle[2][0], 0)
        self.Eye.ppc.spt_angle[2][1] = FNS.thresh_fn(1.5 - self.Eye.ppc.spt_angle[2][0], 0)

        # normalize
        verg = self.Eye.ppc.spt_angle[2]
        rev = FNS.rev_mus(verg, 'gen')
        verg = verg / (verg + rev)
        self.Eye.ppc.spt_angle[2] = verg

        self.Eye.ppc.spt_pres_eye = self.Eye.ppc.spt_angle
        spt_pres_eye = self.Eye.ppc.spt_pres_eye

        # ____________________________________________________________________________________________________________
        # sample compensation for spatial representation of nonfoveated target due to eye movement
        for k in range(3):
            for n in range(2):
                self.Eye.ppc.ltm_bino_targ[k][n] = \
                    RK4.rk4(ltm_bino_targ[k][n], - 0 * dv_bino_targ[k][n] * eye_hold *
                            (-0.0 * ltm_bino_targ[k][n] + binoc_map), step)

        bino_targ_est = np.array([[np.sum(eye_hold * binoc_map * ltm_bino_targ[k][n])
                                   for n in range(2)] for k in range(3)])

        self.Eye.ppc.dv_bino_targ = (1 * bino_targ_est + spt_pres_eye - spt_targ_eye)

        # check learning of compensation for eye movement
        self.Eye.ppc.bino_eye_est = 1 * bino_targ_est + dv_bino_targ

        # update target estimate during learning
        #self.Eye.ppc.spt_targ_eye = spt_targ_eye * (1 - on) + (1 * bino_targ_est + spt_pres_eye) * on

        # update target estimate after learning
        self.Eye.ppc.spt_targ_eye = 1 * bino_targ_est + spt_pres_eye

        # normalize target estimate
        spt_targ_eye = FNS.thresh_fn(self.Eye.ppc.spt_targ_eye, 0)
        rev_eye = FNS.rev_mus(spt_targ_eye, 'spt')
        self.Eye.ppc.spt_targ_eye = spt_targ_eye / (spt_targ_eye + rev_eye + 0.01)

        # ____________________________________________________________________________________________________________
        # sample compensation for spatial representation of nonfoveated target due to head movement
        b_max, a_max = parse_neck
        bound = ptssjnt.ptssjnt_bound(b_max, a_max, num, num)
        for b in bound[0]:
            for a in bound[1]:
                self.Eye.ppc.ltm_targ_head[b][a] = \
                    RK4.rk4(ltm_targ_head[b][a], - 0 * dv_targ_head * head_hold *
                            (-0.0 * ltm_targ_head[b][a] + ptssjnt.ptssjnt_gradient(b, a, b_max, a_max)), step)

                self.Eye.ppc.sptmp_head[b][a] = ptssjnt.ptssjnt_gradient(b, a, b_max, a_max)

        sptmp_head = self.Eye.ppc.sptmp_head
        targ_head_est = np.array([[np.sum(head_hold * sptmp_head * tran_ltm_targ_head[k][n])
                                   for n in range(2)] for k in range(3)])


        self.Eye.ppc.dv_targ_head = 1 * targ_head_est + spt_pres_eye - spt_targ_head

        # check learning of compensation for head movement
        self.Eye.ppc.bino_head_est = 1 * targ_head_est + dv_targ_head

        # update target estimate during learning
        #self.Eye.ppc.spt_targ_head = spt_targ_head * (1 - on) + (1 * targ_head_est + spt_pres_eye) * on

        # update target estimate after learning
        self.Eye.ppc.spt_targ_head = 1 * targ_head_est + spt_pres_eye

        # normalize target estimate
        spt_targ_head = FNS.thresh_fn(self.Eye.ppc.spt_targ_head, 0)
        rev_head = FNS.rev_mus(spt_targ_head, 'spt')
        self.Eye.ppc.spt_targ_head = spt_targ_head / (spt_targ_head + rev_head + 0.01)

        # ____________________________________________________________________________________________________________
        # update target estimate invariant wrt eye movement and head movement
        self.Eye.ppc.spt_targ = (spt_pres_eye + 1 * targ_head_est + 1 * bino_targ_est)

        # normalize target estimate
        spt_targ = FNS.thresh_fn(self.Eye.ppc.spt_targ, 0)
        rev_targ = FNS.rev_mus(spt_targ, 'spt')
        self.Eye.ppc.spt_targ = spt_targ / (spt_targ + rev_targ + 0.01)



    # reset non-learning activity btw trials
    def Reset(self, t, T):
        size = self.ret_size
        num = self.num_deg

        if t % T == 0:
            # retina
            self.Eye.ret.ret_map = np.zeros((2, 2 * size, 2 * size))
            self.Eye.ret.gaze_map = np.zeros((2, 2 * size, 2 * size))

            # colliculus
            self.Eye.sc.top_layer = np.zeros((2, 2 * size, 2 * size))
            self.Eye.sc.bot_layer = np.zeros((2, 2 * size, 2 * size))

            # substantia
            self.Eye.snr.fixate_habit = np.ones((2))
            self.Eye.snr.sc_top = np.ones((2, 2 * size, 2 * size))
            self.Eye.snr.sc_bot = np.ones((2, 2 * size, 2 * size))

            # brainstem
            self.Eye.bs.long_burst = np.zeros((2, 2, 2))
            self.Eye.bs.excit_burst = np.zeros((2, 2, 2))
            self.Eye.bs.inhib_burst = np.zeros((2, 2, 2))
            self.Eye.bs.pauser_eye = np.ones((2))
            self.Eye.bs.tonic_eye = np.zeros((2, 2, 2))
            self.Eye.bs.mus_eye = np.zeros((2, 2, 2))
            self.Eye.bs.norm_eye = 1 / 2 * np.ones((2, 2, 2))
            self.Eye.bs.eyeball = size * np.ones((2, 2))

            # parietal
            self.mtm_eye_sc = np.zeros((2, size, num, 2 * size, 2 * size))
            self.dv_targ_eye = np.zeros((2, 2, 2))
            self.sv_targ_eye = np.zeros((2, 2, 2))

            # for learning representations
            #trial = t // T
            #if trial % 2 == 0:

            # parietal
            self.dv_bino_targ = np.zeros((3, 2))
            self.spt_targ_eye = 0 / 2 * np.ones((3, 2))

            self.dv_targ_head = np.zeros((3, 2))
            self.spt_targ_head = 0 / 2 * np.ones((3, 2))

            self.spt_angle = 1 / 2 * np.ones((3, 2))
            self.spt_pres_eye = 1 / 2 * np.ones((3, 2))
            self.spt_targ = 0 / 2 * np.ones((3, 2))











