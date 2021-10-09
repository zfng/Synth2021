import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from INPUT import TargVar, TargFun
from FUNCS import FNS
from EYES import EyeVar, EyeFun
from BMAP3D import MapVar, MapFun

# ---------------------------------------------------------------------------------------------------------------------
# Eye Module - practice eye movement in 3D with representation

if __name__ == '__main__':
    length = 10000
    size = 20
    num = 2 * size
    interval = 200
    on_hold = 0
    off_hold = 50

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ----------------------------------------------------------------------------------------------------------------
    # initialize variables
    origin = np.array((50, 50, 45))
    limit = np.array((100, 100, 120))
    Var = MapVar(ax, limit, origin, size)

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

    TargVar.CoM = origin
    targ_data = Targ.world_targ_cpt(0, interval)
    target = TargVar.target

    EyeVar = EyeVar(size, num)
    Eye = EyeFun(EyeVar, targ_data)

    Var.target = target
    Var.targ_data = targ_data
    Var.estimate = (TargVar.CoM + TargVar.offset) + FNS().polar_to_cart_3D(np.radians(0), np.radians(0), 10)


    # ==============================================================================================================
    # reuse learning copy

    pkl_eye = open('Eye2040.pickle', 'rb')
    old = pk.load(pkl_eye)
    EyeVar.ppc.ltm_targ_eye = old.ltm_targ_eye
    EyeVar.ppc.ltm_bino_targ = old.ltm_bino_targ
    pkl_eye.close()
    # ================================================================================================================

    def pract(t):

        # update variables
        eye_data = EyeVar.bs.eyeball - size
        axial_data = FNS().column_init()
        uplimb_rot = np.array((FNS().uplimb_init(), FNS().uplimb_init()))
        lowlimb_rot = np.array((FNS().lowlimb_init(), FNS().lowlimb_init()))
        append_data = np.array((uplimb_rot, lowlimb_rot))

        Map = MapFun(eye_data, axial_data, append_data, Var)
        shift = Map.CoM_shift(mode)

        targ_data = Targ.world_targ_cpt(t, interval)
        target = TargVar.target

        Var.target = target
        Var.targ_data = targ_data

        if (t % 20) == 0:
            horz, vert, dist = FNS().conv_targ(EyeVar.ppc.spt_targ_eye)
            Var.estimate = (TargVar.CoM + TargVar.offset) + \
                           FNS().polar_to_cart_3D(np.radians(horz), np.radians(vert), dist)

        Eye.Input = targ_data

        Eye.Reset(t, interval)

        # -------------------------------------------------------------------------------------------------------------
        # process movement and representation
        Eye.Retina()
        Eye.Substantia()
        Eye.Colliculus(t, on_hold, off_hold, interval)
        Eye.Cerebellum()
        Eye.Brainstem()
        Eye.ParietalTarg()
        Eye.ParietalSpat(t, interval)

        # -------------------------------------------------------------------------------------------------------------
        # gather data

        (head_cent, head, head_ahead), (left_eye, right_eye), (left_cent, right_cent), (left_fov, right_fov), \
        (left_targ, right_targ) = Map.head_cpt(shift)
        Map.head_plt(head_cent, head, left_eye, right_eye, left_cent, right_cent, left_fov, right_fov)
        Map.targ_plt(head_cent, head_ahead, left_targ, right_targ)

        est = Var.estimate
        Map.est_plt(est, left_fov, right_fov)

        column, pectoral, pelvic = Map.column_cpt(shift)
        Map.column_plt(column, pectoral, pelvic)

        left_uplimb, right_uplimb = Map.uplimb_cpt(shift)
        Map.uplimb_plt(left_uplimb, right_uplimb)

        left_lowlimb, right_lowlimb = Map.lowlimb_cpt(shift)
        Map.lowlimb_plt(left_lowlimb, right_lowlimb)

        return Var.left_eye, Var.right_eye, Var.left_cent, Var.right_cent, Var.left_fov, Var.right_fov,\
               Var.targ, Var.left_line, Var.right_line, \
               Var.est, Var.left_est, Var.right_est, \
               Var.head, Var.head_cent, Var.column, Var.pelvic, Var.pectoral, Var.column_jnt, Var.CoM, \
               Var.left_uplimb, Var.left_uplimb_jnt, Var.right_uplimb, Var.right_uplimb_jnt, \
               Var.left_lowlimb, Var.left_lowlimb_jnt, Var.right_lowlimb, Var.right_lowlimb_jnt,


    ani = animation.FuncAnimation(fig, pract, frames=length, interval=100, blit=True)

    ax.legend()
    ax.set_title('Eye movement and computation of spatial representation')
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------

