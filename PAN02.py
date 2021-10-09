import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from FUNCS import FNS
from COLM import ColVar, ColFun
from ARMS import ArmVar, ArmFun
from BMAP3D import MapVar, MapFun


# ---------------------------------------------------------------------------------------------------------------------
# Column and Arm Modules - practice head and arm movements simultaneously

if __name__ == '__main__':
    length = 10000
    size = 5
    num = 2 * size

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

    ColVar = ColVar(num)
    Col = ColFun(ColVar)
    insert = Map.col_cpt(shift)[5]
    ColVar.spc.mus_insert = FNS().arrform(insert, 'axial')

    ArmVar = ArmVar(num)
    Arm = ArmFun(ArmVar)
    insert = Map.left_arm_cpt(shift)[5], Map.right_arm_cpt(shift)[5]
    ArmVar.spc.mus_insert = FNS().arrform(insert, 'append')


    # --------------------------------------------------------------------------------------------------------------

    def pract(t):

        # process movement

        Col.ParietalSpat()
        Col.Motor()
        Col.SpinalSkel()
        Col.SpinalCore()
        Col.Cerebellum()


        Arm.ParietalSpat()
        Arm.Motor()
        Arm.SpinalSkel()
        Arm.SpinalCore()
        Arm.Cerebellum()


        # ------------------------------------------------------------------------------------------------------------
        # update variables
        eye_data = FNS().eye_init()
        axial_data = ColVar.spc.ang_plot
        uplimb_rot = ArmVar.spc.ang_plot
        lowlimb_rot = np.array((FNS().lowlimb_init(), FNS().lowlimb_init()))
        append_data = np.array((uplimb_rot, lowlimb_rot))

        Map = MapFun(eye_data, axial_data, append_data, Var)
        shift = Map.CoM_shift(mode)

        insert = Map.col_cpt(shift)[5]
        ColVar.spc.mus_insert = FNS().arrform(insert, 'axial')

        insert = Map.left_arm_cpt(shift)[5], Map.right_arm_cpt(shift)[5]
        ArmVar.spc.mus_insert = FNS().arrform(insert, 'append')

        # -------------------------------------------------------------------------------------------------------------
        # gather data

        (head_cent, head, head_ahead), (left_eye, right_eye), (left_cent, right_cent), (left_fov, right_fov), \
        (left_targ, right_targ) = Map.head_cpt(shift)
        Map.head_plt(head_cent, head, left_eye, right_eye, left_cent, right_cent, left_fov, right_fov)

        column, pectoral, pelvic = Map.column_cpt(shift)
        Map.column_plt(column, pectoral, pelvic)

        left_uplimb, right_uplimb = Map.uplimb_cpt(shift)
        Map.uplimb_plt(left_uplimb, right_uplimb)

        left_lowlimb, right_lowlimb = Map.lowlimb_cpt(shift)
        Map.lowlimb_plt(left_lowlimb, right_lowlimb)

        return Var.left_eye, Var.right_eye, Var.left_cent, Var.right_cent, Var.left_fov, Var.right_fov,\
               Var.head, Var.head_cent, Var.column, Var.pelvic, Var.pectoral, Var.column_jnt, Var.CoM, \
               Var.left_uplimb, Var.left_uplimb_jnt, Var.right_uplimb, Var.right_uplimb_jnt, \
               Var.left_lowlimb, Var.left_lowlimb_jnt, Var.right_lowlimb, Var.right_lowlimb_jnt,


    ani = animation.FuncAnimation(fig, pract, frames=length, interval=100, blit=True)

    ax.set_title('Head movement and arm movement babbling')
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------




