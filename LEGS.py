import numpy as np

from FUNCS import FNS, RK4
from PTSS import PtssJoint as ptssjnt
from PTSS import PtssSpatial as ptssspt


# variable class for Leg Module
class LegVar:
    def __init__(self, num_deg):
        self.num_deg = num_deg

        self.ppc = self.Parietal(num_deg)
        self.mtc = self.Motor()
        self.bs = self.Brainstem()
        self.spc = self.SpinalCord()
        self.cbm = self.Cerebellum()

    class Parietal:
        def __init__(self, num_deg):
            self.num_deg = num_deg

            self.spt_targ = FNS().foot_init()
            self.spt_pres = FNS().foot_init()
            self.tpv = 1 / 2 * np.ones((2, 3, 2, 2))
            self.ppv = 1 / 2 * np.ones((2, 3, 2, 2))

            self.dv_erg = np.zeros((2, 3, 2, 2))
            self.trk_prog = -1
            self.trk_mode = -1
            self.ltm_mode = 0
            self.trk_plot = 0

            self.ltm_mot_spt = np.zeros((2, 3, num_deg, num_deg, 3, 2))
            self.dv_mot_spt = np.zeros((2, 3, 3, 2))
            self.sv_mot_spt = FNS().foot_init()
            self.sptmp_mot_spt = np.zeros((2, 3, num_deg, num_deg))

            self.ltm_spt_mot = np.zeros((2, 3, num_deg, num_deg, num_deg, 2, 2))
            self.dv_spt_mot = np.zeros((2, 3, 2, 2))
            self.sv_spt_mot = np.zeros((2, 3, 2, 2))
            self.sptmp_spt_mot = np.zeros((2, 3, num_deg, num_deg, num_deg))

            self.mot_spt_est = np.zeros((2, 3, 3, 2))
            self.spt_mot_est = np.zeros((2, 3, 2, 2))

    class Motor:
        def __init__(self):
            self.opv = 1 / 2 * np.ones((2, 3, 2, 2))
            self.sfv = 1 / 2 * np.ones((2, 3, 2, 2))
            self.ifv = 1 / 2 * np.ones((2, 3, 2, 2))
            self.ofpv = np.zeros((2, 3, 2, 2))

            ones = np.ones(2)
            zeros = np.zeros(2)
            one_zero = np.array((ones, zeros))
            zero_zero = np.array((zeros, zeros))
            self.left = np.array([(one_zero, zero_zero, zero_zero), (zero_zero, zero_zero, zero_zero)])
            self.right = np.array([(zero_zero, zero_zero, zero_zero), (one_zero, zero_zero, zero_zero)])
            self.speed_mod = 1.0

    class Brainstem:
        def __init__(self):
            self.limb = np.zeros((2))
            self.inter = np.zeros((2))
            self.coeff = np.array([(0.9, 0.55), (0.45, 0.9)])

    class SpinalCord:
        def __init__(self):
            self.alpha_moto = np.zeros((2, 3, 2, 2))
            self.stat_gamma = np.zeros((2, 3, 2, 2))
            self.dynm_gamma = np.zeros((2, 3, 2, 2))
            self.renshaw = np.zeros((2, 3, 2, 2))
            self.ia_int = np.zeros((2, 3, 2, 2))
            self.prim_spin = np.zeros((2, 3, 2, 2))
            self.seco_spin = np.zeros((2, 3, 2, 2))
            self.extra_mus = np.zeros((2, 3, 2, 2))
            self.stat_intra = np.zeros((2, 3, 2, 2))
            self.dynm_intra = np.zeros((2, 3, 2, 2))

            self.equi_leng = np.array([(((8.83, 8.83), (3.74, 3.74)), ((22.65, 22.65), (0, 0)),
                                        ((17.03, 17.26), (0, 0))),
                                       (((8.83, 8.83), (3.74, 3.74)), ((22.65, 22.65), (0, 0)),
                                        ((17.03, 17.26), (0, 0)))])

            self.mus_leng = np.array([(((8.83, 8.83), (3.74, 3.74)), ((22.65, 22.65), (0, 0)),
                                       ((17.03, 17.26), (0, 0))),
                                      (((8.83, 8.83), (3.74, 3.74)), ((22.65, 22.65), (0, 0)),
                                       ((17.03, 17.26), (0, 0)))])

            self.mus_derv = np.zeros((2, 3, 2, 2))
            self.mus_inert = 1 * np.ones((2, 3, 2))
            self.mus_forc = np.zeros((2, 3, 2, 2))
            self.ext_forc = np.zeros((2, 2, 3, 2))

            self.mus_insert = np.zeros((2, 3, 2, 2, 3))

            self.ang_posn = np.zeros((2, 3, 2))
            self.ang_velo = np.zeros((2, 3, 2))
            self.ang_plot = np.zeros((2, 3, 2))

    class Cerebellum:
        def __init__(self):
            self.granule = np.zeros((2, 3, 2, 2))
            self.golgi = np.zeros((2, 3, 2, 2))
            self.purkinje = np.ones((2, 3, 2, 2))
            self.olivary = np.zeros((2, 3, 2, 2))
            self.climb = np.zeros((2, 3, 2, 2))
            self.basket = np.zeros((2, 3, 2, 2))
            self.nuclear = np.zeros((2, 3, 2, 2))
            self.rubral = np.zeros((2, 3, 2, 2))
            self.mtm_purkj = np.ones((2, 3, 2, 2))


# method class for Leg Module
class LegFun:
    def __init__(self, LegVar):
        self.Leg = LegVar
        self.FNS = FNS()
        self.RK4 = RK4()
        self.ste_size = 0.01
        self.num_deg = self.Leg.num_deg
        self.ptssjnt = ptssjnt(self.num_deg, self.num_deg)
        self.ptssspt = ptssspt(self.num_deg, self.num_deg, self.num_deg)

    # model ParietalCortex for learning present representation
    def ParietalMot(self):
        num = self.num_deg
        step = self.ste_size
        ptss = self.ptssjnt
        FNS = self.FNS
        RK4 = self.RK4

        spt_pres = self.Leg.ppc.spt_pres
        ltm = self.Leg.ppc.ltm_mot_spt
        tran_ltm = np.transpose(ltm, (0, 1, 4, 5, 2, 3))
        sv_spt = self.Leg.ppc.sv_mot_spt
        leg = FNS.thresh_fn(self.Leg.ppc.ppv, 0)
        parse = FNS.parse_append(leg, num)

        # reset non-dynamic maps
        self.Leg.ppc.sptmp_mot_spt = np.zeros((2, 3, num, num))

        # sample present representation of limbs
        for s in range(2):
            for l in range(3):
                b_max, a_max = parse[s][l]
                bound = ptss.ptssjnt_bound(b_max, a_max, num, num)

                for b in bound[0]:
                    for a in bound[1]:
                        self.Leg.ppc.ltm_mot_spt[s][l][b][a] = \
                            RK4.rk4(ltm[s][l][b][a], 0 * ptss.ptssjnt_gradient(b, a, b_max, a_max) *
                                    (-0.1 * ltm[s][l][b][a] + spt_pres[s][l]), step)

                        self.Leg.ppc.sptmp_mot_spt[s][l][b][a] = ptss.ptssjnt_gradient(b, a, b_max, a_max)

        sptmp = self.Leg.ppc.sptmp_mot_spt
        pres_est = np.array([[[[np.sum(sptmp[s][l] * tran_ltm[s][l][k][n])
                                for n in range(2)] for k in range(3)] for l in range(3)] for s in range(2)])
        rev_est = np.array([[FNS.rev_mus(pres_est[s][l], 'spt') for l in range(3)] for s in range(2)])
        self.Leg.ppc.sv_mot_spt = pres_est / (pres_est + rev_est + 0.01)

        # check learning of present representation
        self.Leg.ppc.mot_spt_est = sv_spt - spt_pres


    # model ParietalCortex for learning motor command
    def ParietalSpat(self):
        step = self.ste_size
        num = self.num_deg
        ptts = self.ptssspt
        RK4 = self.RK4
        FNS = self.FNS
        leg = self.Leg.ppc.ppv
        opv = FNS.thresh_fn(self.Leg.mtc.opv, 0)
        rev_opv = FNS.rev_mus(opv, 'jnt')
        net_prim = FNS.thresh_fn(FNS.diff_mus(self.Leg.spc.prim_spin, 'append'), 0)
        rev_prim = FNS.rev_mus(net_prim, 'jnt')
        spt_targ = self.Leg.ppc.spt_targ

        dv_mot = self.Leg.ppc.dv_spt_mot
        sv_mot = self.Leg.ppc.sv_spt_mot

        dv_spt = self.Leg.ppc.dv_mot_spt
        spt_pres = self.Leg.ppc.sv_mot_spt

        parse_spt = FNS.parse_targ(dv_spt, num, 'jnt')

        ltm_mot = self.Leg.ppc.ltm_spt_mot
        tran_ltm_mot = np.transpose(ltm_mot, (0, 1, 5, 6, 2, 3, 4))

        targ = self.Leg.ppc.tpv
        pres = leg
        old_prog = self.Leg.ppc.trk_prog
        old_mode = self.Leg.ppc.trk_mode
        erg = self.Leg.ppc.dv_erg

        # reset non-dynamic maps to zero
        self.Leg.ppc.sptmp_spt_mot = np.zeros((2, 3, num, num, num))

        # ____________________________________________________________________________________________________________
        # sample motor command

        """
        self.Leg.ppc.dv_mot_spt = 1 * (spt_targ - spt_pres)

        for s in range(2):
            for l in range(3):
                a_max, b_max, r_max = parse_spt[s][l]

                bound = ptts.ptssspt_bound(a_max, b_max, r_max, num, num, num)

                for a in bound[0]:
                    for b in bound[1]:
                        for r in bound[2]:
                            self.Leg.ppc.ltm_spt_mot[s][l][a][b][r] = \
                                RK4.rk4(ltm_mot[s][l][a][b][r], -0 * dv_mot[s][l] *
                                        (-0.0 * ltm_mot[s][l][a][b][r] +
                                         ptts.ptssspt_gradient(a, b, r, a_max, b_max, r_max)), step)

                            self.Leg.ppc.sptmp_spt_mot[s][l][a][b][r] = \
                                ptts.ptssspt_gradient(a, b, r, a_max, b_max, r_max)

        sptmp = self.Leg.ppc.sptmp_spt_mot
        leg_est = np.array([[[[np.sum(sptmp[s][l] * tran_ltm_mot[s][l][m][n])
                               for n in range(2)] for m in range(2)] for l in range(3)] for s in range(2)])

        self.Leg.ppc.dv_spt_mot = 1 * leg_est + targ - leg

        # check learning of motor command
        self.Leg.ppc.spt_mot_est = leg_est + dv_mot

        # for coordination only
        self.Leg.ppc.tpv = (1 /2 + -1 * leg_est)

        self.Leg.ppc.sv_spt_mot = targ - leg

        self.Leg.ppc.ppv = RK4.rk4(leg,
                                   (1 - leg) * (1 * opv + 1 * rev_prim) - leg * (1 * rev_opv + 1 * net_prim), step)
        """
        # ____________________________________________________________________________________________________________
        # learn limb movement using endogenous random generator

        self.Leg.ppc.sv_spt_mot = targ - leg
        
        self.Leg.ppc.ppv = RK4.rk4(leg,
                                   (1 - leg) * (1 * opv + 1 * rev_prim) - leg * (1 * rev_opv + 1 * net_prim), step)
    
        dv_err = (targ - pres) * \
                 (self.Leg.mtc.left * FNS.delta_fn(old_mode, 0) + self.Leg.mtc.right * FNS.delta_fn(old_mode, 1))
        cond = FNS.cond_fn(dv_err, 0.1)

        # check stepping phase
        if FNS.delta_fn(cond, 1) == 1 and (old_prog == 0 or old_prog == -1):
            new_mode = (old_mode + 1) % 2
            self.Leg.ppc.trk_mode = new_mode

        # check motor program within stepping phase
        if FNS.delta_fn(cond, 1) == 1:
            new_prog = (old_prog + 1) % 2
            self.Leg.ppc.trk_prog = new_prog

            if new_prog == 0:
            
                mode = self.Leg.ppc.trk_mode

                next_cmd = FNS.rand_prog('step')[mode]
                agn_cmd = next_cmd
                ant_cmd = FNS.rand_fix(agn_cmd, 'rev')

                self.Leg.ppc.dv_erg[mode] = agn_cmd
                self.Leg.ppc.dv_erg[(mode + 1) % 2] = ant_cmd

            else:
                self.Leg.ppc.dv_erg = FNS.rev_mus(erg, 'jnt')

            next = self.Leg.ppc.dv_erg
            self.Leg.ppc.tpv = 1 / 2 + next




    # model MotorCortex
    def Motor(self):
        FNS = self.FNS
        RK4 = self.RK4
        step = self.ste_size
        prim_spin = FNS.thresh_fn(self.Leg.spc.prim_spin, 0)
        rev_spin = FNS.rev_mus(prim_spin, 'jnt')
        seco_spin = FNS.thresh_fn(self.Leg.spc.seco_spin, 0)
        net_spin = FNS.thresh_fn(prim_spin - seco_spin, 0)
        rev_net = FNS.rev_mus(net_spin, 'jnt')
        leg = self.Leg.ppc.ppv
        rev_leg = FNS.rev_mus(leg, 'jnt')
        sv_mot = FNS.thresh_fn(self.Leg.ppc.sv_spt_mot, 0)
        rev_sv = FNS.rev_mus(sv_mot, 'jnt')
        opv = FNS.thresh_fn(self.Leg.mtc.opv, 0)
        ofpv = FNS.thresh_fn(self.Leg.mtc.ofpv, 0)
        sfv = FNS.thresh_fn(self.Leg.mtc.sfv, 0)
        ifv = FNS.thresh_fn(self.Leg.mtc.ifv, 0)
        GO = self.Leg.mtc.speed_mod
        nucle = FNS.thresh_fn(self.Leg.cbm.nuclear, 0)
        rev_nucle = FNS.rev_mus(nucle, 'jnt')

        # compute movement command
        self.Leg.mtc.opv = RK4.rk4(opv,
                                   (1 - opv) * (GO * sv_mot + 1 * leg) - opv * (GO * rev_sv + 1 * rev_leg), step)

        self.Leg.mtc.sfv = RK4.rk4(sfv, (1 - sfv) * (1 * prim_spin) - sfv * (1 * rev_spin), step)

        self.Leg.mtc.ifv = RK4.rk4(ifv, (1 - ifv) * (1 * net_spin + nucle) - ifv * (1 * rev_net + rev_nucle), step)

        self.Leg.mtc.ofpv = RK4.rk4(ofpv, -2 * ofpv + (opv + sfv + ifv), step)

    # model Brainstem
    def Brainstem(self):
        step = self.ste_size
        FNS = self.FNS
        RK4 = self.RK4
        mus_spin = FNS.thresh_fn(self.Leg.spc.prim_spin, 0)
        spin_input = FNS.extract_spin(mus_spin)
        limb = FNS.thresh_fn(self.Leg.bs.limb, 0)
        inter = FNS.thresh_fn(self.Leg.bs.inter, 0)
        coeff = self.Leg.bs.coeff
        cos_input = 0.1
        GO = np.max(self.Leg.mtc.speed_mod)

        # compute rhythmic output
        excit_input = 5 * FNS.sigmoid_fn(limb, 0.5, 2.0) + 0.55 * (1 * spin_input + cos_input * GO)

        inhib_input = 3 * np.dot(coeff, FNS.sigmoid_fn(inter, 0.5, 2))

        self.Leg.bs.limb = RK4.rk4(limb, -1 * limb + (1.0 - limb) * excit_input - (2 + limb) * inhib_input, step)
        self.Leg.bs.inter = RK4.rk4(inter, (1 - inter) * limb - inter, step)


    # model SpinalCord for musculoskeletal variables
    def SpinalSkel(self):
        step = self.ste_size
        RK4 = self.RK4
        FNS = self.FNS
        left_insert, right_insert = self.Leg.spc.mus_insert
        ang_velo = self.Leg.spc.ang_velo
        ang_posn = self.Leg.spc.ang_posn
        old_left_len, old_right_len = self.Leg.spc.mus_leng
        equi_leng = self.Leg.spc.equi_leng
        mus_forc = self.Leg.spc.mus_forc
        diff_forc = FNS.cutoff_fn(FNS.diff_force(mus_forc, 'append'), 0.0)
        extra_mus = FNS.thresh_fn(self.Leg.spc.extra_mus, 0)
        inert = self.Leg.spc.mus_inert

        ext_forc = 1 * self.Leg.spc.ext_forc
        diff_ext = FNS.cutoff_fn(ext_forc[0] - ext_forc[1], 0.0)

        # compute joint angles
        self.Leg.spc.ang_velo = RK4.rk4(ang_velo, (1 / inert) * (diff_forc + 1 * diff_ext - 5 * ang_velo), step)
        self.Leg.spc.ang_posn = RK4.rk4(ang_posn, 1 * ang_velo, step)
        self.Leg.spc.ang_plot = np.array([FNS.angle_bound(FNS.bound_fn(1 * ang_posn[s], 1), 'leg') for s in range(2)])

        new_left_pos = left_insert
        new_left_len = FNS.mus_len(new_left_pos, 'append')
        new_left_derv = FNS.mus_derv(old_left_len, new_left_len, step, 'append')

        self.Leg.spc.mus_leng[0] = new_left_len
        self.Leg.spc.mus_derv[0] = new_left_derv

        new_right_pos = right_insert
        new_right_len = FNS.mus_len(new_right_pos, 'append')
        new_right_derv = FNS.mus_derv(old_right_len, new_right_len, step, 'append')

        self.Leg.spc.mus_leng[1] = new_right_len
        self.Leg.spc.mus_derv[1] = new_right_derv

        new_len = np.array((new_left_len, new_right_len))
        self.Leg.spc.mus_forc = FNS.thresh_fn(extra_mus + FNS.leg_mus(new_len - equi_leng), 0)

        # ____________________________________________________________________________________________________________

        # update drawing mode based on state of change of muscle length in response to command
        mus_derv = self.Leg.spc.mus_derv
        derv_input = 1 * FNS.extract_spin(mus_derv)
        self.Leg.ppc.trk_plot = int(1 * np.heaviside(derv_input[0] - derv_input[1], 0) +
                                    0 * np.heaviside(derv_input[1] - derv_input[0], 0))


    # model SpinalCord for neural variables
    def SpinalCore(self):
        """
        # for option "locomotion with subcortical cpg" in PAN03.py
        FNS = self.FNS
        cpg = FNS.thresh_fn(self.Leg.bs.limb, 0)
        rythm_input = 5 * FNS.extract_rythm(cpg)
        ofpv = 0 * FNS.thresh_fn(self.Leg.mtc.ofpv, 0)
        """

        # ____________________________________________________________________________________________________________

        """
        # for option "locomotion with cortical cpg" in PAN03.py
        FNS = self.FNS
        cpg = FNS.thresh_fn(self.Leg.bs.limb, 0)
        rythm_input = 0 * FNS.extract_rythm(cpg)
        ofpv = 1.0 * FNS.thresh_fn(self.Leg.mtc.ofpv, 0)
        """

        # ____________________________________________________________________________________________________________


        # for option "locomotion with cortical and subcortical cpgs" in PAN03.py
        FNS = self.FNS
        cpg = FNS.thresh_fn(self.Leg.bs.limb, 0)
        rythm_input = 1.2 * FNS.extract_rythm(cpg)
        ofpv = 1.2 * FNS.thresh_fn(self.Leg.mtc.ofpv, 0)


        # ____________________________________________________________________________________________________________

        """
        # for PAN05.py
        FNS = self.FNS
        cpg = FNS.thresh_fn(self.Leg.bs.limb, 0)
        rythm_input = 1.3 * FNS.extract_rythm(cpg)
        ofpv = 1.3 * FNS.thresh_fn(self.Leg.mtc.ofpv, 0)
        """

        # ____________________________________________________________________________________________________________
        step = self.ste_size
        FNS = self.FNS
        RK4 = self.RK4
        GO = self.Leg.mtc.speed_mod
        sv_mot = FNS.thresh_fn(self.Leg.ppc.sv_spt_mot, 0)
        rev_sv = FNS.rev_mus(sv_mot, 'jnt')
        opv = self.Leg.mtc.opv
        ia_int = FNS.thresh_fn(self.Leg.spc.ia_int, 0)
        rev_int = FNS.rev_mus(ia_int, 'jnt')
        alpha_moto = FNS.thresh_fn(self.Leg.spc.alpha_moto, 0)
        renshaw = FNS.thresh_fn(self.Leg.spc.renshaw, 0)
        rev_renshaw = FNS.rev_mus(renshaw, 'jnt')
        prim_spin = FNS.thresh_fn(self.Leg.spc.prim_spin, 0)
        seco_spin = FNS.thresh_fn(self.Leg.spc.seco_spin, 0)
        rubral = FNS.thresh_fn(self.Leg.cbm.rubral, 0)
        mus_leng = self.Leg.spc.mus_leng
        mus_derv = self.Leg.spc.mus_derv
        mus_forc = FNS.thresh_fn(self.Leg.spc.mus_forc, 0)
        stat_gamma = FNS.thresh_fn(self.Leg.spc.stat_gamma, 0)
        dynm_gamma = FNS.thresh_fn(self.Leg.spc.dynm_gamma, 0)
        stat_intra = FNS.thresh_fn(self.Leg.spc.stat_intra, 0)
        dynm_intra = FNS.thresh_fn(self.Leg.spc.dynm_intra, 0)
        equi_leng = self.Leg.spc.equi_leng
        stat_input = FNS.thresh_fn(stat_intra + FNS.leg_mus(mus_leng - equi_leng), 0)
        dynm_input = FNS.thresh_fn(FNS.diff_mus(dynm_intra, 'append') + FNS.leg_mus(mus_derv), 0)
        extra_mus = FNS.thresh_fn(self.Leg.spc.extra_mus, 0)
        jnt_fdbk = 1 * FNS.thresh_fn(FNS.jnt_recept(self.Leg.spc.ang_plot, 'leg'), 0)
        rev_jnt = FNS.rev_mus(jnt_fdbk, 'jnt')

        big_size = 1 + 5 * (ofpv + 1 * rythm_input)
        med_size = 0.1 + 0.5 * alpha_moto
        sma_size = 0.01 + 0.05 * (ofpv + 1 * rythm_input)

        # process movement command
        self.Leg.spc.ia_int = \
            RK4.rk4(ia_int, (10 - ia_int) * (ofpv + prim_spin) -
                    ia_int * (1 + renshaw + rev_int), step)

        self.Leg.spc.alpha_moto = \
            RK4.rk4(alpha_moto, (5 * big_size - alpha_moto) * (ofpv + rubral + prim_spin + rev_jnt + 1 * rythm_input) -
                    (alpha_moto + 1) * (0.5 + renshaw + rev_int), step)

        self.Leg.spc.renshaw = \
            RK4.rk4(renshaw, (5 * big_size - renshaw) * (med_size * alpha_moto) -
                    renshaw * (1 + rev_renshaw + 5 * rubral), step)

        self.Leg.spc.extra_mus = \
            RK4.rk4(extra_mus, (big_size - extra_mus) * (sma_size * alpha_moto) -
                    sma_size * extra_mus - mus_forc, step)

        self.Leg.spc.stat_gamma = \
            RK4.rk4(stat_gamma, (2 - stat_gamma) * (1 * opv) -
                    (1 + stat_gamma) * (0.1 + 0.2 * FNS.sigmoid_fn(renshaw, 0.2, 1)), step)

        self.Leg.spc.stat_intra = RK4.rk4(stat_intra, -1 * stat_intra + (2 - stat_intra) * stat_gamma, step)

        self.Leg.spc.dynm_gamma = \
            RK4.rk4(dynm_gamma, (5 - dynm_gamma) * (1 * GO * sv_mot) -
                    (2 + dynm_gamma) * (0.1 + GO * rev_sv + 0.5 * FNS.sigmoid_fn(renshaw, 0.2, 1)), step)

        self.Leg.spc.dynm_intra = RK4.rk4(dynm_intra, -5 * dynm_intra + (2 - dynm_intra) * dynm_gamma, step)

        self.Leg.spc.prim_spin = RK4.rk4(prim_spin, -2 * prim_spin + (1 - prim_spin) * (stat_input + dynm_input), step)

        self.Leg.spc.seco_spin = RK4.rk4(seco_spin, -2 * seco_spin + (1 - seco_spin) * stat_input, step)


    # model Cerebellum
    def Cerebellum(self):
        step = self.ste_size
        FNS = self.FNS
        RK4 = self.RK4
        GO = self.Leg.mtc.speed_mod
        sv_mot = FNS.thresh_fn(self.Leg.ppc.sv_spt_mot, 0)
        granule = FNS.thresh_fn(self.Leg.cbm.granule, 0)
        golgi = FNS.thresh_fn(self.Leg.cbm.golgi, 0)
        basket = FNS.thresh_fn(self.Leg.cbm.basket, 0)
        prim_spin = FNS.thresh_fn(self.Leg.spc.prim_spin, 0)
        seco_spin = FNS.thresh_fn(self.Leg.spc.seco_spin, 0)
        net_spin = FNS.thresh_fn(prim_spin - seco_spin, 0)
        climb = FNS.thresh_fn(self.Leg.cbm.climb, 0)
        olive = FNS.thresh_fn(self.Leg.cbm.olivary, 0)
        purkj = FNS.thresh_fn(self.Leg.cbm.purkinje, 0)
        rev_purkj = FNS.rev_mus(purkj, 'jnt')
        nuclear = FNS.thresh_fn(self.Leg.cbm.nuclear, 0)
        rubral = FNS.thresh_fn(self.Leg.cbm.rubral, 0)
        mtm = self.Leg.cbm.mtm_purkj

        # compute adaptive gains for dynamic force
        self.Leg.cbm.granule = \
            RK4.rk4(granule, -2 * granule + (1 - granule) * (0.1 + 1 * GO * sv_mot) - (0.5 + granule) * golgi, step)


        self.Leg.cbm.golgi = RK4.rk4(golgi, -1 * golgi + (2 - golgi) * (1 * GO * sv_mot * granule), step)

        self.Leg.cbm.basket = RK4.rk4(basket, -1 * basket + (2 - basket) * granule, step)

        self.Leg.cbm.mtm_purkj = RK4.rk4(mtm, 0.01 * granule * ((1 - mtm) - 10 * climb * mtm), step)

        self.Leg.cbm.purkinje = \
            RK4.rk4(purkj, -2 * purkj +
                    (1 - purkj) * (10 * granule * mtm + climb + FNS.sigmoid_fn(purkj, 0.2, 2) + 0.5) -
                    (0.5 + purkj) * (0.5 * rev_purkj + basket), step)

        self.Leg.cbm.climb = \
            RK4.rk4(climb, -climb + (1 - climb) * (10 * climb + net_spin) - (0.5 + climb) * (10 * olive), step)

        self.Leg.cbm.olivary = RK4.rk4(olive, -0.1 * olive + climb, step)

        self.Leg.cbm.nuclear = \
            RK4.rk4(nuclear, -2 * nuclear + (1 - nuclear) * (0.1 + 10 * net_spin) - (0.5 + nuclear) * 2 * purkj, step)

        self.Leg.cbm.rubral = RK4.rk4(rubral, -0.1 * rubral + nuclear, step)



