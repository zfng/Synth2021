import numpy as np

from FUNCS import FNS, RK4
from PTSS import PtssJoint as ptssjnt
from PTSS import PtssSpatial as ptssspt
from PTSS import Ptss as ptss
from EYES import EyeVar, EyeFun


# variable class for Column Module
class ColVar:
    def __init__(self, num_deg):
        self.num_deg = num_deg
        self.ppc = self.Parietal(num_deg)
        self.mtc = self.Motor()
        self.spc = self.SpinalCord()
        self.cbm = self.Cerebellum()

    class Parietal:
        def __init__(self, num_deg):
            self.num_deg = num_deg

            self.spt_targ = 1 / 2 * np.ones((2, 3, 2))
            self.spt_pres = 1 / 2 * np.ones((2, 3, 2))
            self.tpv = 1 / 2 * np.ones((2, 2, 2))
            self.ppv = 1 / 2 * np.ones((2, 2, 2))

            self.dv_erg = np.zeros((2, 2, 2))
            self.trk_prog = -1

            self.ltm_mot_spt = np.zeros((2, num_deg, num_deg, 3, 2))
            self.dv_mot_spt = np.zeros((2, 3, 2))
            self.sv_mot_spt = 1 / 2 * np.ones((2, 3, 2))
            self.sptmp_mot_spt = np.zeros((2, num_deg, num_deg))

            self.ltm_spt_mot = np.zeros((2, num_deg, num_deg, num_deg, 2, 2))
            self.dv_spt_mot = np.zeros((2, 2, 2))
            self.sv_spt_mot = np.zeros((2, 2, 2))
            self.sptmp_spt_mot = np.zeros((2, num_deg, num_deg, num_deg))

            self.mot_spt_est = np.zeros((2, 3, 2))
            self.spt_mot_est = np.zeros((2, 2, 2))

    class Motor:
        def __init__(self):
            self.opv = 1 / 2 * np.ones((2, 2, 2))
            self.sfv = 1 / 2 * np.ones((2, 2, 2))
            self.ifv = 1 / 2 * np.ones((2, 2, 2))
            self.ofpv = np.zeros((2, 2, 2))

            ones = np.ones(2)
            zeros = np.zeros(2)
            one_zero = np.array((ones, zeros))
            weights = np.array([one_zero, one_zero])
            self.weights = weights
            self.speed_mod = 1

    class SpinalCord:
        def __init__(self):
            self.alpha_moto = np.zeros((2, 2, 2))
            self.stat_gamma = np.zeros((2, 2, 2))
            self.dynm_gamma = np.zeros((2, 2, 2))
            self.renshaw = np.zeros((2, 2, 2))
            self.ia_int = np.zeros((2, 2, 2))
            self.prim_spin = np.zeros((2, 2, 2))
            self.seco_spin = np.zeros((2, 2, 2))
            self.extra_mus = np.zeros((2, 2, 2))
            self.stat_intra = np.zeros((2, 2, 2))
            self.dynm_intra = np.zeros((2, 2, 2))

            self.equi_leng = np.array([((15.52, 15.52), (15.52, 15.52)), ((12.53, 12.53), (10.20, 10.20))])
            self.mus_leng = np.array([((15.52, 15.52), (15.52, 15.52)), ((12.53, 12.53), (10.20, 10.20))])
            self.mus_derv = np.zeros((2, 2, 2))
            self.mus_inert = 1 * np.ones((2, 2))
            self.mus_forc = np.zeros((2, 2, 2))
            self.ext_forc = np.zeros((2, 2, 2))

            self.mus_insert = np.zeros((2, 2, 2, 3))

            self.ang_posn = np.zeros((2, 2))
            self.ang_velo = np.zeros((2, 2))
            self.ang_plot = np.zeros((2, 2))

    class Cerebellum:
        def __init__(self):
            self.granule = np.zeros((2, 2, 2))
            self.golgi = np.zeros((2, 2, 2))
            self.purkinje = np.ones((2, 2, 2))
            self.olivary = np.zeros((2, 2, 2))
            self.climb = np.zeros((2, 2, 2))
            self.basket = np.zeros((2, 2, 2))
            self.nuclear = np.zeros((2, 2, 2))
            self.rubral = np.zeros((2, 2, 2))
            self.mtm_purkj = np.ones((2, 2, 2))


# method class for Column Module
class ColFun:
    def __init__(self, ColVar):
        self.Col = ColVar
        self.FNS = FNS()
        self.RK4 = RK4()
        self.ste_size = 0.01
        self.num_deg = self.Col.num_deg
        self.ptssspt = ptssspt(self.num_deg, self.num_deg, self.num_deg)
        self.ptssjnt = ptssjnt(self.num_deg, self.num_deg)
        self.ptss = ptss(self.num_deg, self.num_deg)

    # model ParietalCortex for learning present representation
    def ParietalMot(self):
        num = self.num_deg
        step = self.ste_size
        ptss = self.ptssjnt
        FNS = self.FNS
        RK4 = self.RK4

        spt_pres = self.Col.ppc.spt_pres
        ltm = self.Col.ppc.ltm_mot_spt
        tran_ltm = np.transpose(ltm, (0, 3, 4, 1, 2))
        sv_spt = self.Col.ppc.sv_mot_spt
        col = self.Col.ppc.ppv
        parse = FNS.parse_axial(col, num)

        # reset non-dynamic maps to zero
        self.Col.ppc.sptmp_mot_spt = np.zeros((2, num, num))

        # sample present representation of column
        for l in range(2):
            b_max, a_max = parse[l]
            bound = ptss.ptssjnt_bound(b_max, a_max, num, num)

            for b in bound[0]:
                for a in bound[1]:
                    self.Col.ppc.ltm_mot_spt[l][b][a] = \
                        RK4.rk4(ltm[l][b][a], 0 * ptss.ptssjnt_gradient(b, a, b_max, a_max) *
                                (-0.1 * ltm[l][b][a] + spt_pres[l]), step)

                    self.Col.ppc.sptmp_mot_spt[l][b][a] = ptss.ptssjnt_gradient(b, a, b_max, a_max)

        sptmp = self.Col.ppc.sptmp_mot_spt
        pres_est = np.array([[[np.sum(sptmp[l] * tran_ltm[l][k][n])
                               for n in range(2)] for k in range(3)] for l in range(2)])
        rev_est = np.array([FNS.rev_mus(pres_est[l], 'spt') for l in range(2)])
        self.Col.ppc.sv_mot_spt = pres_est / (pres_est + rev_est + 0.01)

        # check learning of present representation
        self.Col.ppc.mot_spt_est = sv_spt - spt_pres

    # model ParietalCortex for learning motor command
    def ParietalSpat(self):
        step = self.ste_size
        num = self.num_deg
        ptts = self.ptssspt
        RK4 = self.RK4
        FNS = self.FNS
        col = self.Col.ppc.ppv
        opv = self.Col.mtc.opv
        rev_opv = FNS.rev_mus(opv, 'col')
        net_prim = FNS.thresh_fn(FNS.diff_mus(self.Col.spc.prim_spin, 'axial'), 0.0)
        rev_prim = FNS.rev_mus(net_prim, 'col')
        spt_targ = self.Col.ppc.spt_targ

        dv_mot = self.Col.ppc.dv_spt_mot
        sv_mot = self.Col.ppc.sv_spt_mot

        dv_spt = self.Col.ppc.dv_mot_spt
        spt_pres = self.Col.ppc.sv_mot_spt

        parse_spt = FNS.parse_targ(dv_spt, num, 'col')

        ltm_mot = self.Col.ppc.ltm_spt_mot
        tran_ltm_mot = np.transpose(ltm_mot, (0, 4, 5, 1, 2, 3))

        targ = self.Col.ppc.tpv
        pres = col
        old_prog = self.Col.ppc.trk_prog
        erg = self.Col.ppc.dv_erg


        # reset non-dynamic maps to zero
        self.Col.ppc.sptmp_spt_mot = np.zeros((2, num, num, num))

        # ____________________________________________________________________________________________________________
        # sample motor command

        """
        self.Col.ppc.dv_mot_spt = spt_targ - spt_pres
        
        for l in range(2):
            a_max, b_max, r_max = parse_spt[l]
            bound = ptts.ptssspt_bound(a_max, b_max, r_max, num, num, num)

            for a in bound[0]:
                for b in bound[1]:
                    for r in bound[2]:
                        self.Col.ppc.ltm_spt_mot[l][a][b][r] = \
                            RK4.rk4(ltm_mot[l][a][b][r], -0 * dv_mot[l] *
                                    (-0.0 * ltm_mot[l][a][b][r] +
                                     ptts.ptssspt_gradient(a, b, r, a_max, b_max, r_max)), step)

                        self.Col.ppc.sptmp_spt_mot[l][a][b][r] = ptts.ptssspt_gradient(a, b, r, a_max, b_max, r_max)

        sptmp = self.Col.ppc.sptmp_spt_mot
        col_est = np.array([[[np.sum(sptmp[l] * tran_ltm_mot[l][m][n])
                              for n in range(2)] for m in range(2)] for l in range(2)])

        self.Col.ppc.dv_spt_mot = 1 * col_est + targ - col

        # check learning of motor command
        self.Col.ppc.spt_mot_est = col_est + dv_mot

        # for coordination only
        self.Col.ppc.tpv = (1 / 2 + -1 * col_est)

        self.Col.ppc.sv_spt_mot = targ - col


        self.Col.ppc.ppv = RK4.rk4(col,
                                   (1 - col) * (1 * opv + 1 * rev_prim) - col * (1 * rev_opv + 1 * net_prim), step)
        """
        # ____________________________________________________________________________________________________________
        # learn limb movement using endogenous random generator


        self.Col.ppc.sv_spt_mot = targ - col
        
        self.Col.ppc.ppv = RK4.rk4(col,
                                   (1 - col) * (1 * opv + 1 * rev_prim) - col * (1 * rev_opv + 1 * net_prim), step)


        dv_err = (targ - pres) * self.Col.mtc.weights
        cond = FNS.cond_fn(dv_err[0], 0.1)

        # check motor program
        if FNS.delta_fn(cond, 1) == 1:
            new_prog = (old_prog + 1) % 2
            self.Col.ppc.trk_prog = new_prog

            if new_prog == 0:

                self.Col.ppc.dv_erg = FNS.rand_prog('col')

            else:
                self.Col.ppc.dv_erg = FNS.rev_mus(erg, 'col')
    
            next = self.Col.ppc.dv_erg
            self.Col.ppc.tpv = 1/2 + next


    # model MotorCortex
    def Motor(self):
        FNS = self.FNS
        RK4 = self.RK4
        step = self.ste_size
        prim_spin = FNS.thresh_fn(self.Col.spc.prim_spin, 0)
        rev_spin = FNS.rev_mus(prim_spin, 'col')
        seco_spin = FNS.thresh_fn(self.Col.spc.seco_spin, 0)
        net_spin = FNS.thresh_fn(prim_spin - seco_spin, 0)
        rev_net = FNS.rev_mus(net_spin, 'col')
        col = self.Col.ppc.ppv
        rev_col = FNS.rev_mus(col, 'col')
        sv_mot = FNS.thresh_fn(self.Col.ppc.sv_spt_mot, 0.0)
        rev_sv = FNS.rev_mus(sv_mot, 'col')
        opv = FNS.thresh_fn(self.Col.mtc.opv, 0)
        ofpv = FNS.thresh_fn(self.Col.mtc.ofpv, 0)
        sfv = FNS.thresh_fn(self.Col.mtc.sfv, 0)
        ifv = FNS.thresh_fn(self.Col.mtc.ifv, 0)
        nucle = FNS.thresh_fn(self.Col.cbm.nuclear, 0)
        rev_nucle = FNS.rev_mus(nucle, 'col')
        GO = self.Col.mtc.speed_mod

        # compute movement command
        self.Col.mtc.opv = RK4.rk4(opv,
                                   (1 - opv) * (GO * sv_mot + 1 * col) - opv * (GO * rev_sv + 1 * rev_col), step)

        self.Col.mtc.sfv = RK4.rk4(sfv, (1 - sfv) * (1 * prim_spin) - sfv * (1 * rev_spin), step)

        self.Col.mtc.ifv = RK4.rk4(ifv, (1 - ifv) * (1 * net_spin + nucle) - ifv * (1 * rev_net + rev_nucle), step)

        self.Col.mtc.ofpv = RK4.rk4(ofpv, -2 * ofpv + (opv + sfv + ifv), step)


    # model SpinalCord for musculoskeletal variables
    def SpinalSkel(self):
        step = self.ste_size
        RK4 = self.RK4
        FNS = self.FNS
        neck_insert, pelvic_insert = self.Col.spc.mus_insert
        ang_velo = self.Col.spc.ang_velo
        ang_posn = self.Col.spc.ang_posn
        old_neck_len, old_pelvic_len = self.Col.spc.mus_leng
        equi_leng = self.Col.spc.equi_leng
        mus_forc = self.Col.spc.mus_forc
        diff_forc = FNS.cutoff_fn(FNS.diff_force(mus_forc, 'axial'), 0.0)
        extra_mus = FNS.thresh_fn(self.Col.spc.extra_mus, 0)
        inert = self.Col.spc.mus_inert

        # compute joint angles
        self.Col.spc.ang_velo = RK4.rk4(ang_velo, (1 / inert) * (diff_forc - 5 * ang_velo), step)
        self.Col.spc.ang_posn = RK4.rk4(ang_posn, 1 * ang_velo, step)
        self.Col.spc.ang_plot = FNS.angle_bound(FNS.bound_fn(1 * ang_posn, 1), 'col')

        new_neck_pos = neck_insert
        new_neck_len = FNS.mus_len(new_neck_pos, 'axial')
        new_neck_derv = FNS.mus_derv(old_neck_len, new_neck_len, step, 'axial')

        self.Col.spc.mus_leng[0] = new_neck_len
        self.Col.spc.mus_derv[0] = new_neck_derv

        new_pelvic_pos = pelvic_insert
        new_pelvic_len = FNS.mus_len(new_pelvic_pos, 'axial')
        new_pelvic_derv = FNS.mus_derv(old_pelvic_len, new_pelvic_len, step, 'axial')

        self.Col.spc.mus_leng[1] = new_pelvic_len
        self.Col.spc.mus_derv[1] = new_pelvic_derv

        new_len = np.array((new_neck_len, new_pelvic_len))
        self.Col.spc.mus_forc = FNS.thresh_fn(1 * extra_mus + 1 * FNS.col_mus(new_len - equi_leng), 0)


    # model SpinalCord for neural variables
    def SpinalCore(self):

        #for PAN02.py
        FNS = self.FNS
        ofpv = 1 * FNS.thresh_fn(self.Col.mtc.ofpv, 0)

        # ____________________________________________________________________________________________________________

        step = self.ste_size
        FNS = self.FNS
        RK4 = self.RK4
        GO = self.Col.mtc.speed_mod
        sv_mot = FNS.thresh_fn(self.Col.ppc.sv_spt_mot, 0.0)
        rev_sv = FNS.rev_mus(sv_mot, 'col')
        opv = self.Col.mtc.opv
        ia_int = FNS.thresh_fn(self.Col.spc.ia_int, 0)
        rev_ia_int = FNS.rev_mus(ia_int, 'col')
        alpha_moto = FNS.thresh_fn(self.Col.spc.alpha_moto, 0)
        renshaw = FNS.thresh_fn(self.Col.spc.renshaw, 0)
        rev_renshaw = FNS.rev_mus(renshaw, 'col')
        prim_spin = FNS.thresh_fn(self.Col.spc.prim_spin, 0)
        seco_spin = FNS.thresh_fn(self.Col.spc.seco_spin, 0)
        rubral = FNS.thresh_fn(self.Col.cbm.rubral, 0)
        mus_leng = self.Col.spc.mus_leng
        mus_derv = self.Col.spc.mus_derv
        mus_forc = FNS.thresh_fn(self.Col.spc.mus_forc, 0.0)
        stat_gamma = FNS.thresh_fn(self.Col.spc.stat_gamma, 0)
        dynm_gamma = FNS.thresh_fn(self.Col.spc.dynm_gamma, 0)
        stat_intra = FNS.thresh_fn(self.Col.spc.stat_intra, 0)
        dynm_intra = FNS.thresh_fn(self.Col.spc.dynm_intra, 0)
        equi_leng = self.Col.spc.equi_leng
        stat_input = FNS.thresh_fn(stat_intra + 1 * FNS.col_mus(mus_leng - equi_leng), 0)
        dynm_input = FNS.thresh_fn(FNS.diff_mus(dynm_intra, 'axial') + 1 * FNS.col_mus(mus_derv), 0)
        extra_mus = FNS.thresh_fn(self.Col.spc.extra_mus, 0)

        big_size = 1 + 5 * ofpv
        med_size = 0.1 + 0.5 * alpha_moto
        sma_size = 0.01 + 0.05 * ofpv

        # process movement command
        self.Col.spc.ia_int = \
         RK4.rk4(ia_int, 0.5 * (10 - ia_int) * (ofpv + prim_spin) -
                 ia_int * (1 + renshaw + rev_ia_int), step)

        self.Col.spc.alpha_moto = \
         RK4.rk4(alpha_moto, (5 * big_size - alpha_moto) * (ofpv + rubral + prim_spin) -
                 (alpha_moto + 1) * (0.5 + renshaw + rev_ia_int), step)

        self.Col.spc.renshaw = \
         RK4.rk4(renshaw, (5 * big_size - renshaw) * (med_size * alpha_moto) -
                 renshaw * (1 + rev_renshaw + 5 * rubral), step)

        self.Col.spc.extra_mus = \
         RK4.rk4(extra_mus, (big_size - extra_mus) * (sma_size * alpha_moto) -
                 sma_size * extra_mus - mus_forc, step)

        self.Col.spc.stat_gamma = \
         RK4.rk4(stat_gamma, (2 - stat_gamma) * (1.0 * opv) -
                 (1 + stat_gamma) * (0.1 + 0.2 * FNS.sigmoid_fn(renshaw, 0.2, 1)), step)

        self.Col.spc.stat_intra = RK4.rk4(stat_intra, -1 * stat_intra + (2 - stat_intra) * stat_gamma, step)

        self.Col.spc.dynm_gamma = \
         RK4.rk4(dynm_gamma, (5 - dynm_gamma) * (1 * GO * sv_mot) -
                 (2 + dynm_gamma) * (0.1 + GO * rev_sv + 0.5 * FNS.sigmoid_fn(renshaw, 0.2, 1)), step)

        self.Col.spc.dynm_intra = RK4.rk4(dynm_intra, -5 * dynm_intra + (2 - dynm_intra) * dynm_gamma, step)

        self.Col.spc.prim_spin = RK4.rk4(prim_spin, -2 * prim_spin + (1 - prim_spin) * (stat_input + dynm_input), step)

        self.Col.spc.seco_spin = RK4.rk4(seco_spin, -2 * seco_spin + (1 - seco_spin) * stat_input, step)

        # model Cerebellum
    def Cerebellum(self):
         step = self.ste_size
         FNS = self.FNS
         RK4 = self.RK4
         GO = self.Col.mtc.speed_mod
         sv_mot = self.Col.ppc.sv_spt_mot
         granule = FNS.thresh_fn(self.Col.cbm.granule, 0)
         golgi = FNS.thresh_fn(self.Col.cbm.golgi, 0)
         basket = FNS.thresh_fn(self.Col.cbm.basket, 0)
         prim_spin = FNS.thresh_fn(self.Col.spc.prim_spin, 0)
         seco_spin = FNS.thresh_fn(self.Col.spc.seco_spin, 0)
         net_spin = FNS.thresh_fn(prim_spin - seco_spin, 0)
         climb = FNS.thresh_fn(self.Col.cbm.climb, 0)
         olive = FNS.thresh_fn(self.Col.cbm.olivary, 0)
         purkj = FNS.thresh_fn(self.Col.cbm.purkinje, 0)
         rev_purkj = FNS.rev_mus(purkj, 'col')
         nuclear = FNS.thresh_fn(self.Col.cbm.nuclear, 0)
         rubral = FNS.thresh_fn(self.Col.cbm.rubral, 0)
         mtm = self.Col.cbm.mtm_purkj

         self.Col.cbm.granule = \
             RK4.rk4(granule, -2 * granule + (1 - granule) * (0.1 + 1 * GO * sv_mot) - (0.5 + granule) * golgi, step)

         self.Col.cbm.golgi = RK4.rk4(golgi, -golgi + (2 - golgi) * (1 * GO * sv_mot * granule), step)

         self.Col.cbm.basket = RK4.rk4(basket, -basket + (2 - basket) * granule, step)

         self.Col.cbm.mtm_purkj = RK4.rk4(mtm, 0.01 * granule * ((1 - mtm) - 10 * climb * mtm), step)

         self.Col.cbm.purkinje = \
             RK4.rk4(purkj, -2 * purkj +
                     (1 - purkj) * (10 * granule * mtm + 1 * climb + FNS.sigmoid_fn(purkj, 0.2, 2) + 0.5) -
                     (0.5 + purkj) * (0.5 * rev_purkj + 1 * basket), step)

         self.Col.cbm.climb = \
             RK4.rk4(climb, -climb + (1 - climb) * (climb + 10 * net_spin) - (0.5 + climb) * (10 * olive), step)

         self.Col.cbm.olivary = RK4.rk4(olive, -0.1 * olive + climb, step)


         self.Col.cbm.nuclear = \
             RK4.rk4(nuclear, -2 * nuclear + (1 - nuclear) * (0.1 + 10 * net_spin) - (0.5 + nuclear) * 2 * purkj, step)

         self.Col.cbm.rubral = RK4.rk4(rubral, -0.1 * rubral + nuclear, step)







