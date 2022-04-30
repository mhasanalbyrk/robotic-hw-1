import string
from typing import List

import keyboard
import vpython as vpy
import numpy as np
from numpy import ndarray, radians

origin_vector: vpy.vector = vpy.vector(0, 0, 0)
base_frame_x = vpy.vector(1, 0, 0)
base_frame_y = vpy.vector(0, 1, 0)
base_frame_z = vpy.vector(0, 0, 1)

base_frame_curve_x = vpy.arrow(length=2, pos=origin_vector, axis=base_frame_x, color=vpy.color.cyan)
base_frame_curve_y = vpy.arrow(length=2, pos=origin_vector, axis=base_frame_y, color=vpy.color.cyan)
base_frame_curve_z = vpy.arrow(length=2, pos=origin_vector, axis=base_frame_z, color=vpy.color.cyan)

base_x_label = vpy.label(text="x",
                         color=vpy.color.red,
                         pos=base_frame_curve_x.pos + base_frame_curve_x.axis + vpy.vector(0, 0, 0),
                         box=False)
base_y_label = vpy.label(text="y",
                         color=vpy.color.red,
                         pos=base_frame_curve_y.pos + base_frame_curve_y.axis + vpy.vector(0, 0, 0),
                         box=False)
base_z_label = vpy.label(text="z",
                         color=vpy.color.red,
                         pos=base_frame_curve_z.pos + base_frame_curve_z.axis + vpy.vector(0, 0, 0),
                         box=False)

# motor_1: vpy.cylinder = vpy.cylinder(pos=vpy.vector(0, 0, 0), radius=0.3, axis=vpy.vector(0, 0, 1));
# rod1_1: vpy.cylinder = vpy.cylinder(pos=vpy.vector(0, 0, 1), radius=0.2, axis=vpy.vector(0, 0, 1.5))
# part_1: vpy.compound = vpy.compound([motor_1, rod1_1])
# part_1.color = vpy.color.red
#
# motor_2: vpy.cylinder = vpy.cylinder(pos=vpy.vector(0, -1.5, 0), radius=0.3, axis=vpy.vector(0, 3, 0))
# rod2_1: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x, -motor_2.pos.y, motor_2.pos.z), radius=0.2,
#                                     axis=vpy.vector(3, 0, 0))
# rod2_2: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x, motor_2.pos.y, motor_2.pos.z), radius=0.2,
#                                     axis=vpy.vector(3, 0, 0))
# rod2_3: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + 3, motor_2.pos.y, motor_2.pos.z), radius=0.2,
#                                     axis=vpy.vector(0, 3, 0))
# rod2_4: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + 3, motor_2.pos.y + 1.5, motor_2.pos.z), radius=0.2,
#                                     axis=vpy.vector(3, 0, 0))
#
# part_2: vpy.compound = vpy.compound([motor_2, rod2_4, rod2_3, rod2_1, rod2_2], pos=vpy.vector(3, 0, 2.5))
# part_2.color = vpy.color.blue
#
#
# part_3: vpy.compound = part_2.clone(pos=vpy.vector(part_2.pos.x + 6, part_2.pos.y, part_2.pos.z))
# part_3.color = vpy.color.green
# part_2.rotate(angle=np.radians(-90), origin=vpy.vector(0, 0, 2.5), axis=vpy.vector(0, 0, 1))
# part_3.rotate(angle=vpy.radians(30), axis=vpy.vector(0, 1, 0))
# part_2.rotate(angle=vpy.radians(-30), axis=vpy.vector(0, 1, 0), origin=vpy.vector(0, 0, 2.5))

THETA_ARR = [0, 0, 0]
THETA_1: int = 0
THETA_2: int = 0
THETA_3: int = 0

A_1 = 2.5
A_2 = 6
A_3 = 6

r_1_2 = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]])


def rotation_matrix(degrees: int, axis: string) -> ndarray:
    theta_degrees: radians = np.radians(degrees)

    if axis == 'x':
        # print('x rotation matrix')
        x = np.array([[0, np.cos(theta_degrees), -np.sin(theta_degrees)],
                      [0, np.sin(theta_degrees), np.cos(theta_degrees)],
                      [1, 0, 0]])
        return x

    if axis == 'y':
        # print('y rotation matrix')
        y = np.array([[np.cos(theta_degrees), 0, np.sin(theta_degrees)],
                      [0, 1, 0],
                      [-np.sin(theta_degrees), 0, np.cos(theta_degrees)]])
        return y
    if axis == 'z':
        # print('z rotation matrix')
        z = np.array([[np.cos(theta_degrees), -np.sin(theta_degrees), 0],
                      [np.sin(theta_degrees), np.cos(theta_degrees), 0],
                      [0, 0, 1]])
        return z


def redraw(matrix: ndarray, part: vpy.cylinder, origin: vpy.vector, part_number, rotated):
    rot: ndarray[3][3]
    rot = matrix[0:3]
    rot = rot[:, [0, 1, 2]]
    print(f'pos before for part {part_number} = {part.pos}')
    part.pos = vpy.vector(part_1.pos.x + matrix[0][3], part_1.pos.y + matrix[1][3], matrix[2][3])

    print(f'pos after for part {part_number} = {part.pos}')
    print(f'part.axis before for part {part_number} = {part.axis}')

    new_x_axis = np.matmul(rot, np.array([
        part_1.axis.x,
        part_1.axis.y,
        part_1.axis.z
    ]))
    if rotated == 1:
        new_x_axis = np.matmul(rot, np.array([
            part_1.axis.x,
            part_1.axis.y,
            part_1.axis.z
        ]))

        if part_number == 2:
            # turning first part, what happens to second
            eye = np.eye(3)

            part.axis = vpy.vector(new_x_axis[0], new_x_axis[1], new_x_axis[2])
            part.axis = part.axis * 3
            norm = part.axis.norm()
            big_norm = norm * 3
            rod2_1.axis = big_norm
            rod2_2.axis = big_norm
            rod2_3.axis = big_norm
            rod2_4.axis = big_norm

            rod2_1.pos.x = 1.5 * np.cos(np.radians(THETA_1 + 90))
            rod2_1.pos.y = 1.5 * np.sin(np.radians(THETA_1 + 90))
            rod2_1.pos.z = part.pos.z
            rod2_2.pos.x = -1.5 * np.cos(np.radians(THETA_1 + 90))
            rod2_2.pos.y = -1.5 * np.sin(np.radians(THETA_1 + 90))
            rod2_2.pos.z = part.pos.z
            rod2_3.pos.x = part.pos.x + big_norm.x
            rod2_3.pos.y = part.pos.y + big_norm.y
            rod2_3.pos.z = part.pos.z + big_norm.z
            rod2_4.pos = rod2_3.pos

            # rod2_2.axis = norm * 3
            # rod2_3.axis = norm * 3
            # rod2_4.axis = norm * 3
            # rod2_1.axis.y = part.axis.y * 3

        elif part_number == 3:
            part.axis = vpy.vector(new_x_axis[0], new_x_axis[1], new_x_axis[2])
            part.axis = part.axis * 3
            norm = part.axis.norm()
            big_norm = norm * 3

            rod3_1.axis = big_norm
            rod3_2.axis = big_norm
            rod3_3.axis = big_norm
            a = np.array([[np.cos(np.radians(THETA_1)), -np.sin(np.radians(THETA_1))],
                          [np.sin(np.radians(THETA_1)), np.cos(np.radians(THETA_1))]])
            b = a @ np.array([[6],
                              [1.5]])

            # rod3_1.pos.x = 6 * np.cos(np.radians(THETA_1)) - 1.5 * np.sin(np.radians(THETA_1))
            # rod3_1.pos.y = 6 * np.sin(np.radians(THETA_1)) + 1.5 * np.cos(np.radians(THETA_1))
            rod3_1.pos.x = b[0][0]
            rod3_1.pos.y = b[1][0]
            b = a @ np.array([[6],
                              [-1.5]])
            rod3_2.pos.x = b[0][0]
            rod3_2.pos.y = b[1][0]
            rod3_1.pos.z = part.pos.z
            rod3_2.pos.z = part.pos.z
            rod3_3.pos.x = part.pos.x + big_norm.x
            rod3_3.pos.y = part.pos.y + big_norm.y
            rod3_3.pos.z = part.pos.z + big_norm.z
            rod3_4.axis = big_norm
            rod3_4.pos = rod3_3.pos




    elif rotated == 2:
        print(f'rot for part2 = {rot}')
        if part_number == 2:


        elif part_number == 3:
            part.axis = vpy.vector(float(str(new_x_axis[0])[:2]), float(str(new_x_axis[1])[:2]),
                                   float(str(new_x_axis[2])[:2]))


    # new_y_axis = np.matmul(rot, np.array([part.axis.x, part.axis.y, part.axis.z]))
    # new_z_axis = np.matmul(rot, np.array([part.axis.x, part.axis.y, part.axis.z]))

    elif rotated == 3:
        print(f'rot for part3= {rot}')
        rot = np.matmul(rot, rotation_matrix(THETA_3, 'z'))
        new_x_axis = np.matmul(rot, np.array([
            part_1.axis.x,
            part_1.axis.y,
            part_1.axis.z
        ]))
        # part.pos.x = matrix[0][3] +3
        # part.pos.y = matrix[1][3]
        # part.pos.z = matrix[2][3]
        # part.rotate(angle=np.radians(-5), axis=vpy.vector(0, 0, 1), origin=origin)

        part.axis = vpy.vector(new_x_axis[0], new_x_axis[1], new_x_axis[2])

    # end_x_label.pos = part_2.pos + vpy.vector(offset, 0, 0)
    # part_3_pos.pos = part_3.pos + vpy.vector(offset, 0, 0)

    # print(f' new axis by rot matrix x {new_x_axis}')
    # part.axis = vpy.vector(new_x_axis[0], new_x_axis[1], new_x_axis[2])
    # print(part.origin)
    print(f'part.axis after for part {part_number} = {part.axis}')


def full_homo_matrix(part_number: int) -> ndarray:
    if part_number == 2:
        return calc_homo_matrix(1)
    if part_number == 3:
        return np.matmul(calc_homo_matrix(1), calc_homo_matrix(2))


def calc_homo_matrix(part_number: int) -> ndarray:
    fix = np.array([
        [0, 0, 0, 1]
    ])
    rot_mat: ndarray[3][3]

    if part_number == 1:  # calculates H_1_2 matrix
        rot_mat = np.matmul(rotation_matrix(THETA_1, 'z'), r_1_2)
        d_mat = np.array([
            [0],
            [0],
            [A_1]
        ])
        temp = np.column_stack([rot_mat, d_mat])
        return np.row_stack([temp, fix])
    elif part_number == 2:  # calculates H_2_3 matrix
        rot_mat = rotation_matrix(THETA_2, 'z')
        d_mat = np.array([
            [A_2 * np.cos(THETA_2)],
            [A_2 * np.sin(THETA_2)],
            [0]
        ])
        temp = np.column_stack([rot_mat, d_mat])
        return np.row_stack([temp, fix])


if __name__ == "__main__":
    offset = 0.0
    THETA_UNIT = 5
    motor_1: vpy.cylinder = vpy.cylinder(pos=vpy.vector(0, 0, 0), radius=0.3, axis=vpy.vector(0, 0, 1));
    rod1_1: vpy.cylinder = vpy.cylinder(pos=vpy.vector(0, 0, 1), radius=0.2, axis=vpy.vector(0, 0, 1.5))
    part_1: vpy.compound = vpy.compound([motor_1, rod1_1])
    part_1.color = vpy.color.red
    original_motor_1_axis: List = [0, 3, 0]
    original_motor_2_axis: List = [0, 3, 0]
    motor_2: vpy.cylinder = vpy.cylinder(pos=vpy.vector(0, -1.5, A_1), radius=0.3, axis=vpy.vector(0, 3, 0))
    motor_2.color = vpy.color.green
    rod2_1: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x, -motor_2.pos.y, motor_2.pos.z), radius=0.2,
                                        axis=vpy.vector(3, 0, 0))
    rod2_1.color = vpy.color.cyan
    rod2_2: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x, motor_2.pos.y, motor_2.pos.z), radius=0.2,
                                        axis=vpy.vector(3, 0, 0))
    rod2_2.color = vpy.color.magenta
    rod2_3: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + 3, motor_2.pos.y, motor_2.pos.z), radius=0.2,
                                        axis=vpy.vector(0, 3, 0))
    rod2_3.color = vpy.color.yellow

    rod2_4: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + 3, motor_2.pos.y + 1.5, motor_2.pos.z),
                                        radius=0.2,
                                        axis=vpy.vector(3, 0, 0))

    motor_3: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + 6, -1.5, A_1), radius=0.3,
                                         axis=vpy.vector(0, 3, 0))
    motor_3.color = vpy.color.blue

    rod3_1: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + 6, -motor_2.pos.y, motor_3.pos.z), radius=0.2,
                                        axis=vpy.vector(3, 0, 0))
    rod3_1.color = vpy.color.cyan

    rod3_2: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + 6, motor_2.pos.y, motor_3.pos.z), radius=0.2,
                                        axis=vpy.vector(3, 0, 0))
    rod3_2.color = vpy.color.magenta

    rod3_3: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + 6 + 3, motor_2.pos.y, motor_3.pos.z), radius=0.2,
                                        axis=vpy.vector(0, 3, 0))
    rod3_3.color = vpy.color.yellow

    rod3_4: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + 6 + 3, motor_2.pos.y + 1.5, motor_3.pos.z),
                                        radius=0.2,
                                        axis=vpy.vector(3, 0, 0))
    motor_3 = vpy.compound([motor_3], origin=vpy.vector(6, 0, 2.5))
    rod3_3 = vpy.compound([rod3_3], origin=vpy.vector(9, 0, 2.5))
    motor_2 = vpy.compound([motor_2], origin=vpy.vector(0, 0, 2.5))
    rod2_3 = vpy.compound([rod2_3], origin=vpy.vector(3, 0, 2.5))

    # part_2: vpy.compound = vpy.compound([motor_2, rod2_4, rod2_3, rod2_1, rod2_2],
    #                                     origin=vpy.vector(0, 0, 2.5), axis=motor_2.axis)

    # part_2.color = vpy.color.blue
    # part_2.pos.z += 2.5
    # part_3: vpy.compound = vpy.compound([motor_3, rod3_4, rod3_3, rod3_1, rod3_2],
    #                                     origin=vpy.vector(6, 0, 2.5), axis=motor_3.axis)
    # part_3.color = vpy.color.green
    # part_3.pos.x = part_2.pos.x + 3
    # part_3.pos.z = part_2.pos.z

    end_x_label = vpy.label(text="PART_2 POS",
                            color=vpy.color.cyan,
                            pos=motor_2.pos + vpy.vector(0.025, 0, 0),
                            box=False)
    part_3_pos = vpy.label(text="PART_3 POS",
                           color=vpy.color.green,
                           pos=motor_3.pos + vpy.vector(0.025, 0, 0),
                           box=False)

    while True:
        vpy.rate(5)
        try:
            if keyboard.is_pressed('a'):
                THETA_1 -= THETA_UNIT
                if THETA_1 + 360 == 0:
                    THETA_1 = 0

                print(f'Theta 1 = {THETA_1}')
                homogeneous_matrix = full_homo_matrix(2)
                print(f'Homo matrix for 1 to 2 = {homogeneous_matrix}')

                # redraw(homogeneous_matrix, motor_2,
                #        vpy.vector(homogeneous_matrix[0][3], homogeneous_matrix[1][3], homogeneous_matrix[2][3]))
                redraw(homogeneous_matrix, motor_2, vpy.vector(0, 0, 2.5), part_number=2, rotated=1)

                homogeneous_matrix = full_homo_matrix(3)
                print(f'Homo matrix for 1 to 3 = {homogeneous_matrix}')

                redraw(homogeneous_matrix, motor_3,
                       vpy.vector(0, 0, 2.5), part_number=3, rotated=1)
                # redraw(homogeneous_matrix, motor_3, vpy.vector(6 * np.cos(THETA_2), 6 * np.sin(THETA_2), 2.5))

            elif keyboard.is_pressed('s'):
                THETA_1 += THETA_UNIT
                if THETA_1 + 360 == 0:
                    THETA_1 = 0

                print(f'Theta 1 = {THETA_1}')
                homogeneous_matrix = full_homo_matrix(2)
                # print(f'Homo matrix for 1 to 2 = {homogeneous_matrix}')

                # redraw(homogeneous_matrix, motor_2,
                #        vpy.vector(homogeneous_matrix[0][3], homogeneous_matrix[1][3], homogeneous_matrix[2][3]))
                redraw(homogeneous_matrix, motor_2, vpy.vector(0, 0, 2.5), part_number=2, rotated=1)

                homogeneous_matrix = full_homo_matrix(3)
                # redraw(homogeneous_matrix, part_3,
                #        vpy.vector(0, 0, 2.5), part_number=3, rotated=1)
            # 
            # 
            elif keyboard.is_pressed('q'):
                THETA_2 -= THETA_UNIT
                if THETA_2 + 360 == 0:
                    THETA_2 = 0

                print(f'Theta 2 = {THETA_2}')
                homogeneous_matrix = full_homo_matrix(2)
                print(f'Homo matrix for 1 to 2 = {homogeneous_matrix}')

                # redraw(homogeneous_matrix, part_2,
                #        vpy.vector(homogeneous_matrix[0][3], homogeneous_matrix[1][3], homogeneous_matrix[2][3]))
                # redraw(homogeneous_matrix, part_2, vpy.vector(0, 0, 2.5), part_number=2, rotated=2)

                homogeneous_matrix = full_homo_matrix(3)
                print(f'Homo matrix for 1 to 3 = {homogeneous_matrix}')

                # redraw(homogeneous_matrix, part_3,
                #        vpy.vector(0, 0, 2.5), part_number=3, rotated=2)

            # 
            # elif keyboard.is_pressed('w'):
            #     print('Rotating joint 2 10 degrees clock-wise')
            #     homogeneous_matrix = calc_homo_matrix(THETA_UNIT, 1, )
            #     re_draw_new_frame()
            # 
            # 
            elif keyboard.is_pressed('z'):
                THETA_3 -= THETA_UNIT
                if THETA_3 + 360 == 0:
                    THETA_3 = 0

                print(f'Theta 3 = {THETA_3}')
                homogeneous_matrix = full_homo_matrix(3)
                print(f'Homo matrix for 1 to 3 = {homogeneous_matrix}')

            #     redraw(homogeneous_matrix, part_3,
            #            vpy.vector(0, 0, 2.5), part_number=3, rotated=3)
            # #
            # #
            # elif keyboard.is_pressed('x'):
            #     print('Rotating joint 4 10 degrees clock-wise')
            #     homogeneous_matrix = calc_homo_matrix(THETA_UNIT, 3)
            #     re_draw_new_frame(end_frame_curve_x, end_frame_curve_y, end_frame_curve_z, homogeneous_matrix)


        except Exception as e:
            print(e.with_traceback())
