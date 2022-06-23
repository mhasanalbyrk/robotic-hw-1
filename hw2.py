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
global H01
H12 = None
H23 = None
H34 = None
H45 = None
H56 = None
A_1 = 2.5
A_2 = 6
A_3 = 6
a4 = 1
a5 = 1
a6 = 1

r_1_2 = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]])
r01 = np.array([[1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]])
r12 = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
r23 = np.array([[0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]])
r34 = np.array([[1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]])
r45 = np.array([[1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]])
r56 = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])


def calculateHomoMatricies(theta1, theta2, theta3, theta4, theta5, theta6):
    global H01
    global H12
    global H23
    global H34
    global H45
    global H56
    R01 = rotation_matrix(theta1) @ r01
    R12 = rotation_matrix(theta2) @ r12
    R23 = rotation_matrix(theta3) @ r23
    R34 = rotation_matrix(theta4) @ r34
    R45 = rotation_matrix(theta5) @ r45
    R56 = rotation_matrix(theta6) @ r56

    d01 = np.array([[0],
                    [0],
                    [A_1]])
    d12 = np.array([[A_2 * np.cos(theta2)],
                    [A_2 * np.sin(theta2)],
                    [0]])
    d23 = np.array([[A_3 * np.cos(theta3)],
                    [A_3 * np.sin(theta3)],
                    [0]])
    d34 = np.array([[0],
                    [0],
                    [a4]])
    d45 = np.array([[a5 * np.cos(theta5)],
                    [a5 * np.sin(theta5)],
                    [0]])
    d56 = np.array([[0],
                    [0],
                    [a6]])

    d34 = np.array([[0],
                    [0],
                    [0]])
    d45 = np.array([[0],
                    [0],
                    [0]])
    d56 = np.array([[0],
                    [0],
                    [0]])
    H01 = np.row_stack([np.column_stack([R01, d01]), np.array([0, 0, 0, 1])])
    H12 = np.row_stack([np.column_stack([R12, d12]), np.array([0, 0, 0, 1])])
    H23 = np.row_stack([np.column_stack([R23, d23]), np.array([0, 0, 0, 1])])
    H34 = np.row_stack([np.column_stack([R34, d34]), np.array([0, 0, 0, 1])])
    H45 = np.row_stack([np.column_stack([R45, d45]), np.array([0, 0, 0, 1])])
    H56 = np.row_stack([np.column_stack([R56, d56]), np.array([0, 0, 0, 1])])


def rotation_matrix(degrees: int, axis: string = 'z') -> ndarray:
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
    fix = np.array([0, 0, 0, 1])
    print(f'pos before for part {part_number} = {part.pos}')
    part.pos = vpy.vector(matrix[0][3], matrix[1][3], matrix[2][3])

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
            theta_2_rot_matrix = rotation_matrix(THETA_2, 'z')
            d_2_1 = np.array([[1.5 * np.sin(np.radians(THETA_1))],
                              [1.5 * np.cos(np.radians(THETA_1))],
                              [0]])
            rod_2_1_homo = np.c_[theta_2_rot_matrix, d_2_1]
            rod_2_1_homo = np.row_stack([rod_2_1_homo, fix])
            myHomo = matrix @ rod_2_1_homo
            trans = myHomo[0:3, 3]
            myRot = myHomo[0:3, 0:3]
            rot = rot @ rotation_matrix(THETA_2, 'z')
            new_x_axis_by_Theta_2 = np.matmul(myRot, np.array([
                part_1.axis.x,
                part_1.axis.y,
                part_1.axis.z
            ]))
            d_2_3_Theta_1 = np.array([[3 * np.cos(np.radians(THETA_1))],
                                      [3 * np.sin(np.radians(THETA_1))],
                                      [0]])
            rod_2_3_Theta_1_homo = np.c_[theta_2_rot_matrix, d_2_3_Theta_1]
            rod_2_3_Theta_1_homo = np.row_stack([rod_2_3_Theta_1_homo, fix])

            d_2_3_Theta_2 = np.array([[3 * np.cos(np.radians(THETA_2))],
                                      [0],
                                      [3 * np.sin(np.radians(THETA_2))]])
            rod_2_3_Theta_2_homo = np.c_[theta_2_rot_matrix, d_2_3_Theta_2]
            rod_2_3_Theta_2_homo = np.row_stack([rod_2_3_Theta_2_homo, fix])

            myHomo_2_3 = matrix @ rod_2_3_Theta_2_homo

            part.axis = vpy.vector(new_x_axis[0], new_x_axis[1], new_x_axis[2])
            part.axis = part.axis * 3
            norm = vpy.vector(new_x_axis_by_Theta_2[0], new_x_axis_by_Theta_2[1], new_x_axis_by_Theta_2[2]).norm()
            big_norm = norm * 3
            rod2_1.axis = big_norm
            rod2_2.axis = big_norm
            rod2_3.axis = part.axis
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


        elif part_number == 3:
            part.pos = rod2_4.pos + rod2_4.axis
            norm = vpy.vector(new_x_axis[0], new_x_axis[1], new_x_axis[2]).norm()
            part.axis = norm * 3

            rot = rot @ rotation_matrix(THETA_2, 'z')
            # new_x_axis_by_Theta_2 = np.matmul(rot, np.array([
            #     part_1.axis.x,
            #     part_1.axis.y,
            #     part_1.axis.z
            # ]))
            # part.axis = vpy.vector(new_x_axis_by_Theta_2[0], new_x_axis_by_Theta_2[1], new_x_axis_by_Theta_2[2])
            # part.axis = part.axis * 3
            norm = part.axis.norm()
            big_norm = norm * 3
            #
            rod3_1.axis = big_norm
            rod3_2.axis = big_norm
            rod3_3.axis = big_norm
            a = np.array([[np.cos(np.radians(THETA_1)), -np.sin(np.radians(THETA_1))],
                          [np.sin(np.radians(THETA_1)), np.cos(np.radians(THETA_1))]])

            b = a @ np.array([[6],
                              [1.5]])

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
        if part_number == 2:
            theta_2_rot_matrix = rotation_matrix(THETA_2, 'z')
            d = np.array([[1.5 * np.sin(np.radians(THETA_1))],
                          [1.5 * np.cos(np.radians(THETA_1))],
                          [0]])
            rod_2_1_homo = np.c_[theta_2_rot_matrix, d]
            rod_2_1_homo = np.row_stack([rod_2_1_homo, fix])
            myHomo = matrix @ rod_2_1_homo
            trans = myHomo[0:3, 3]
            myRot = myHomo[0:3, 0:3]
            rot = rot @ rotation_matrix(THETA_2, 'z')
            new_x_axis_by_Theta_2 = np.matmul(myRot, np.array([
                part_1.axis.x,
                part_1.axis.y,
                part_1.axis.z
            ]))
            rot = rot @ rotation_matrix(THETA_2, 'z')
            new_x_axis = np.matmul(rot, np.array([
                part_1.axis.x,
                part_1.axis.y,
                part_1.axis.z
            ]))
            print(f'rot for part2 = {rot}')

            k = vpy.vector(new_x_axis[0], new_x_axis[1], new_x_axis[2])
            # part.axis = part.axis * 3
            norm = k.norm()
            big_norm = norm * 3
            rod2_1.axis = big_norm
            rod2_2.axis = big_norm
            rod2_3.axis = part.axis
            rod2_4.axis = big_norm
            rod2_3.pos.x = part.pos.x + big_norm.x
            rod2_3.pos.y = part.pos.y + big_norm.y
            rod2_3.pos.z = part.pos.z + big_norm.z
            rod2_4.pos = rod2_3.pos


        elif part_number == 3:
            # norm = vpy.vector(new_x_axis[0], new_x_axis[1], new_x_axis[2]).norm()
            # part.axis = norm * 3

            part.pos = rod2_4.pos + rod2_4.axis

            rot = rot @ rotation_matrix(THETA_3, 'z')
            # new_x_axis_by_Theta_3 = np.matmul(rot, np.array([
            #     part_1.axis.x,
            #     part_1.axis.y,
            #     part_1.axis.z
            # ]))
            #
            # big_norm = vpy.vector(new_x_axis_by_Theta_3[0], new_x_axis_by_Theta_3[1],
            #                       new_x_axis_by_Theta_3[1]).norm() * 3
            # part.axis = big_norm
            # part.pos = rod2_4.pos + rod2_4.axis
            #
            # rod3_1.pos.x = rod2_1.axis.x + 2*big_norm.x
            # rod3_1.pos.y = rod2_1.axis.y + 2*big_norm.y
            # rod3_1.pos.z = rod2_1.axis.z + 2*big_norm.z
            # rod3_1.axis = big_norm
            #
            # rod3_2.axis = big_norm
            # rod3_3.axis = big_norm
            # rod3_4.axis = big_norm



    elif rotated == 3:
        part.pos = rod2_4.pos + rod2_4.axis

        norm = vpy.vector(new_x_axis[0], new_x_axis[1], new_x_axis[2]).norm()
        part.axis = norm * 3
        print(f'rot for part3= {rot}')
        rot = np.matmul(rot, rotation_matrix(THETA_3, 'z'))
        new_x_axis = np.matmul(rot, np.array([
            part_1.axis.x,
            part_1.axis.y,
            part_1.axis.z
        ]))

        big_norm = vpy.vector(new_x_axis[0], new_x_axis[1], new_x_axis[2]).norm() * 3
        rod3_2.axis = big_norm
        rod3_3.axis = big_norm
        rod3_4.axis = big_norm
        rod3_1.axis = big_norm
        rod3_3.pos = motor_3.pos + rod3_2.axis
        rod3_4.pos = motor_3.pos + rod3_2.axis
        # part.rotate(angle=np.radians(-5), axis=vpy.vector(0, 0, 1), origin=origin)

    print(f'part.axis after for part {part_number} = {part.axis}')


def full_homo_matrix(part_number: int) -> ndarray:
    if part_number == 2:
        return calc_homo_matrix(1)
    elif part_number == 3:
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


def calculateJacobian():
    H02 = H01 @ H12
    H03 = H02 @ H23
    H04 = H03 @ H34
    H05 = H04 @ H45
    H06 = H05 @ H56
    eye = np.eye(3)

    jacobianUpper = np.cross((eye @ np.array([[0],
                                              [0],
                                              [1]]).flatten()), H06[0:3, 3].reshape([1, 3]).flatten())
    jacobianUpper = np.column_stack([jacobianUpper, np.cross((H01[0:3, 0:3] @ np.array([[0],
                                                                                        [0],
                                                                                        [1]]).flatten()),
                                                             H06[0:3, 3].reshape([1, 3]) - H01[0:3, 3].reshape(
                                                                 [1, 3])).reshape([3, 1])])
    jacobianUpper = np.column_stack([jacobianUpper, np.cross((H02[0:3, 0:3] @ np.array([[0],
                                                                                        [0],
                                                                                        [1]]).flatten()),
                                                             H06[0:3, 3].reshape([1, 3]) - H02[0:3, 3].reshape(
                                                                 [1, 3])).reshape([3, 1])])
    jacobianUpper = np.column_stack([jacobianUpper, np.cross(H03[0:3, 0:3] @ np.array([[0],
                                                                                       [0],
                                                                                       [1]]).flatten(),
                                                             H06[0:3, 3].reshape([1, 3]) - H03[0:3, 3].reshape(
                                                                 [1, 3])).reshape([3, 1])])
    jacobianUpper = np.column_stack([jacobianUpper, np.cross((H04[0:3, 0:3] @ np.array([[0],
                                                                                        [0],
                                                                                        [1]]).flatten()),
                                                             H06[0:3, 3].reshape([1, 3]) - H04[0:3, 3].reshape(
                                                                 [1, 3])).reshape([3, 1])])
    jacobianUpper = np.column_stack([jacobianUpper, np.cross((H05[0:3, 0:3] @ np.array([[0],
                                                                                        [0],
                                                                                        [1]]).flatten()),
                                                             H06[0:3, 3].reshape([1, 3]) - H05[0:3, 3].reshape(
                                                                 [1, 3])).reshape([3, 1])])

    jacobianLower = np.array([[0],
                              [0],
                              [1]])
    jacobianLower = np.column_stack([jacobianLower, H01[0:3, 0:3] @ np.array([[0],
                                                                              [0],
                                                                              [1]])])
    jacobianLower = np.column_stack([jacobianLower, H02[0:3, 0:3] @ np.array([[0],
                                                                              [0],
                                                                              [1]])])
    jacobianLower = np.column_stack([jacobianLower, H03[0:3, 0:3] @ np.array([[0],
                                                                              [0],
                                                                              [1]])])
    jacobianLower = np.column_stack([jacobianLower, H04[0:3, 0:3] @ np.array([[0],
                                                                              [0],
                                                                              [1]])])
    jacobianLower = np.column_stack([jacobianLower, H05[0:3, 0:3] @ np.array([[0],
                                                                              [0],
                                                                              [1]])])

    return np.row_stack([jacobianUpper, jacobianLower])


if __name__ == "__main__":
    mybox = vpy.box(pos=vpy.vector(13, 0, 0),
                length=1, height=10, width=5)

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

    thetas = np.array([
        [THETA_1],
        [THETA_2],
        [THETA_3],
        [0],
        [0],
        [0]])
    calculateHomoMatricies(thetas[0][0], thetas[1][0], thetas[2][0], thetas[3][0], thetas[4][0],
                           thetas[5][0])
    jacobian = calculateJacobian()

    xDot = 0
    yDot = 100
    zDot = 0
    wX = 0
    wY = 0
    wZ = 0
    movVars = np.array([[xDot],
                        [yDot],
                        [zDot],
                        [wX],
                        [wY],
                        [wZ]])

    while(True):
        vpy.rate(5)
        try:
                calculateHomoMatricies(thetas[0][0], thetas[1][0], thetas[2][0], thetas[3][0], thetas[4][0],
                                       thetas[5][0])
                jacobian = calculateJacobian()
                deltas = np.linalg.pinv(jacobian) @ movVars
                THETA_1 = THETA_1 + deltas[0][0]

                thetas = np.array([
                    [THETA_1],
                    [THETA_2],
                    [THETA_3],
                    [0],
                    [0],
                    [0]])
                calculateHomoMatricies(thetas[0][0], thetas[1][0], thetas[2][0], thetas[3][0], thetas[4][0],
                                       thetas[5][0])
                jacobian = calculateJacobian()

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
                if np.abs(THETA_1) > 30:
                    yDot = -1 * yDot

                movVars = np.array([[xDot],
                                    [yDot],
                                    [zDot],
                                    [wX],
                                    [wY],
                                    [wZ]])
                # redraw(homogeneous_matrix, motor_3, vpy.vector(6 * np.cos(THETA_2), 6 * np.sin(THETA_2), 2.5))

        except Exception as e:
            print(e.with_traceback())
