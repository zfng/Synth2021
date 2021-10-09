import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from FUNCS import FNS
from PTSS import Ptss as ptts
from EYES import EyeVar, EyeFun
from COLM import ColVar, ColFun
from LEGS import LegVar, LegFun
from BMAP3D import MapVar, MapFun
from INPUT import TargVar, TargFun


# --------------------------------------------------------------------------------------------------------------------
# Leg Module - perform eye-leg coordination assisted by eye-head coordination


if __name__ == '__main__':
    length = 5000
    size = 20
    num = 2 * size
    interval = 1000

    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(projection='3d')

    # -----------------------------------------------------------------------------------------------------------------
    # initialize variables

    origin = np.array((20, 50, 45))
    limit = np.array((100, 100, 120))
    Var = MapVar(ax3d, limit, origin, size)

    eye_data = FNS().eye_init()
    axial_data = FNS().column_init()
    uplimb_rot = np.array((FNS().uplimb_init(), FNS().uplimb_init()))
    lowlimb_rot = np.array((FNS().lowlimb_init(), FNS().lowlimb_init()))
    append_data = np.array((uplimb_rot, lowlimb_rot))

    Map = MapFun(eye_data, axial_data, append_data, Var)
    mode = (0, 0)
    shift = Map.CoM_shift(mode)

    TargVar = TargVar(size)
    Targ = TargFun(TargVar)

    TargVar.head_ang = np.zeros(2)
    TargVar.CoM = Var.center
    targ_data = np.zeros((2, 2))
    target = TargVar.target

    EyeVar = EyeVar(size, num)
    Eye = EyeFun(EyeVar, targ_data)

    ColVar = ColVar(num)
    Col = ColFun(ColVar)
    insert = Map.col_cpt(shift)[5]
    ColVar.spc.mus_insert = FNS().arrform(insert, 'axial')

    LegVar = LegVar(num)
    Leg = LegFun(LegVar)
    insert = Map.left_leg_cpt(shift)[5], Map.right_leg_cpt(shift)[5]
    LegVar.spc.mus_insert = FNS().arrform(insert, 'append')

    Var.target = target
    Var.targ_data = targ_data

    # ================================================================================================================
    # reuse learning copy

    pkl_eye = open('Eye2040.pickle', 'rb')
    old = pk.load(pkl_eye)
    EyeVar.ppc.ltm_bino_targ = old.ltm_bino_targ
    EyeVar.ppc.ltm_targ_head = old.ltm_targ_head
    pkl_eye.close()

    pkl_col = open('Col2040.pickle', 'rb')
    old = pk.load(pkl_col)
    ColVar.ppc.ltm_spt_mot = old.ltm_spt_mot
    ColVar.ppc.ltm_mot_spt = old.ltm_mot_spt
    pkl_col.close()

    pkl_leg = open('Leg2040.pickle', 'rb')
    old = pk.load(pkl_leg)
    LegVar.ppc.ltm_spt_mot = old.ltm_spt_mot
    LegVar.ppc.ltm_mot_spt = old.ltm_mot_spt
    pkl_leg.close()
    # ================================================================================================================

    def pract(t):

        parity = -1 * FNS().delta_fn((t // interval) % 2, 0) + 1 * FNS().delta_fn((t // interval) % 2, 1)

        # ----------------------------------------------------------------------------------------------------------
        # process motor command for head
        if t % interval == 0:
            trial = t // interval
            if (trial % 2) == 0:
                input = FNS().rand_prog('init')
                input[0][1] = abs(input[0][1]) * np.array((-1, 1))
                col_posn = 1/ 2 + input
                ColVar.ppc.tpv = col_posn
                ColVar.ppc.ppv = col_posn
                EyeVar.ppc.ppv_neck = col_posn
                col_ang = FNS().diff_posn(col_posn, 'col')
                ColVar.spc.ang_plot = col_ang

            else:
                input = FNS().rand_prog('init')
                input[0][1] = abs(input[0][1]) * np.array((1, -1))
                col_posn = 1 / 2 + input
                ColVar.ppc.tpv = col_posn
                ColVar.ppc.ppv = col_posn
                EyeVar.ppc.ppv_neck = col_posn
                col_ang = FNS().diff_posn(col_posn, 'col')
                ColVar.spc.ang_plot = col_ang

        TargVar.head_ang = ColVar.spc.ang_plot[0]
        TargVar.CoM = Var.center
        targ_data = Targ.world_leg_cpt(t, interval, parity)
        target = TargVar.target

        Var.target = target
        Var.targ_data = targ_data

        Eye.Input = targ_data

        # fixate eyes on target
        left_targ, right_targ = Eye.Input
        left = ptts(num, num).ret_mus_tran(*left_targ)
        right = ptts(num, num).ret_mus_tran(*right_targ)
        EyeVar.bs.norm_eye = np.array((left, right))
        EyeVar.bs.eyeball = FNS().parse_eye(EyeVar.bs.norm_eye, size)
        col_posn = ColVar.ppc.ppv
        EyeVar.ppc.ppv_neck = col_posn

        Eye.Retina()
        Eye.ParietalSpat(t, interval)

        # ----------------------------------------------------------------------------------------------------------
        # process motor command for legs

        if FNS().forwd_period(t, interval / 2, interval) == 1:

            LegVar.ppc.spt_targ = np.array([[EyeVar.ppc.spt_targ for l in range(3)] for s in range(2)])

            GO = 1.5 * np.ones((2, 3, 2, 2))
            LegVar.mtc.speed_mod = GO

            old_mode = LegVar.ppc.trk_plot

            Leg.ParietalMot()
            Leg.ParietalSpat()
            Leg.Motor()
            Leg.Brainstem()
            Leg.SpinalSkel()
            Leg.SpinalCore()
            Leg.Cerebellum()

            new_mode = LegVar.ppc.trk_plot

            trk_change = abs(new_mode - old_mode)

            if trk_change != 0:
                center = Var.center
                Var.origin = center

            Var.trk_change = trk_change

        if FNS().backw_period(t, interval / 2, interval) == 1:

            GO = 1.5 * np.ones((2, 3, 2, 2))
            LegVar.mtc.speed_mod = GO

            old_mode = LegVar.ppc.trk_plot

            LegVar.ppc.dv_mot_spt = np.zeros((2, 3, 3, 2))

            Leg.ParietalSpat()

            LegVar.ppc.tpv = LegVar.ppc.ppv + 1 * (1 / 2 * np.ones((2, 3, 2, 2)) - LegVar.ppc.ppv)

            Leg.Motor()
            Leg.Brainstem()
            Leg.SpinalSkel()
            Leg.SpinalCore()
            Leg.Cerebellum()

            new_mode = LegVar.ppc.trk_plot

            trk_change = abs(new_mode - old_mode)

            if trk_change != 0:
                center = Var.center
                Var.origin = center

            Var.trk_change = trk_change

        # ----------------------------------------------------------------------------------------------------------

        # update variables in ax3d
        eye_data = EyeVar.bs.eyeball - size
        axial_data = ColVar.spc.ang_plot
        uplimb_rot = np.array((FNS().uplimb_init(), FNS().uplimb_init()))
        lowlimb_rot = LegVar.spc.ang_plot
        append_data = np.array((uplimb_rot, lowlimb_rot))

        Map = MapFun(eye_data, axial_data, append_data, Var)
        new_mode = LegVar.ppc.trk_plot
        mode = FNS().conv_mode(new_mode)
        shift = Map.CoM_shift(mode)

        insert = Map.col_cpt(shift)[5]
        ColVar.spc.mus_insert = FNS().arrform(insert, 'axial')

        Map.ext_forc(shift)
        LegVar.spc.ext_forc = np.array((Var.grf, Var.fof))

        insert = Map.left_leg_cpt(shift)[5], Map.right_leg_cpt(shift)[5]
        LegVar.spc.mus_insert = FNS().arrform(insert, 'append')

        # -------------------------------------------------------------------------------------------------------------
        # gather data
        (head_cent, head, head_ahead), (left_eye, right_eye), (left_cent, right_cent), (left_fov, right_fov), \
        (left_targ, right_targ) = Map.head_cpt(shift)
        Map.head_plt(head_cent, head, left_eye, right_eye, left_cent, right_cent, left_fov, right_fov)
        Map.targ_plt(head_cent, head_ahead, left_targ, right_targ)

        column, pectoral, pelvic = Map.column_cpt(shift)
        Map.column_plt(column, pectoral, pelvic)

        left_uplimb, right_uplimb = Map.uplimb_cpt(shift)
        Map.uplimb_plt(left_uplimb, right_uplimb)

        left_lowlimb, right_lowlimb = Map.lowlimb_cpt(shift)
        Map.lowlimb_plt(left_lowlimb, right_lowlimb)

        return Var.left_eye, Var.right_eye, Var.left_cent, Var.right_cent, Var.left_fov, Var.right_fov, \
               Var.targ, Var.targ_line, Var.left_line, Var.right_line, Var.cent_line, \
               Var.head, Var.head_cent, Var.column, Var.pelvic, Var.pectoral, Var.column_jnt, Var.CoM, \
               Var.left_uplimb, Var.left_uplimb_jnt, Var.right_uplimb, Var.right_uplimb_jnt, \
               Var.left_lowlimb, Var.left_lowlimb_jnt, Var.right_lowlimb, Var.right_lowlimb_jnt, \


    ani = animation.FuncAnimation(fig3d, pract, frames=length, interval=100, blit=True)

    ax3d.set_title("Locomotion by visual guidance")
    plt.show()