import keyboard
import numpy as np
import vpython as vpy
from numpy import ndarray, radians
from vpython import vector, color, rate

origin_vector: vector = vector(0, 0, 0)

R_1_2: ndarray = np.array([[1, 0, 0],
                           [0, 0, -1],
                           [0, 1, 0]])

R_3_4: ndarray = np.array([[1, 0, 0],
                           [0, 0, 1],
                           [0, -1, 0]])
R_2_3: ndarray = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]
                           ])
R_4_5: ndarray = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]
                           ])


def rotation_matrix(degrees, axis) -> ndarray:
    theta_degrees: radians = np.radians(degrees)

    if axis == 'x':
        print('x rotation matrix')
        x = np.array([[0, np.cos(theta_degrees), -np.sin(theta_degrees)],
                      [0, np.sin(theta_degrees), np.cos(theta_degrees)],
                      [1, 0, 0]])
        return x

    if axis == 'y':
        print('y rotation matrix')
        y = np.array([[np.cos(theta_degrees), 0, np.sin(theta_degrees)],
                      [0, 1, 0],
                      [-np.sin(theta_degrees), 0, np.cos(theta_degrees)]])
        return y
    if axis == 'z':
        print('z rotation matrix')
        z = np.array([[np.cos(theta_degrees), -np.sin(theta_degrees), 0],
                      [np.sin(theta_degrees), np.cos(theta_degrees), 0],
                      [0, 0, 1]])
        return z


def rotate(theta, joint_number, clock_wise: bool = False) -> ndarray:
    if clock_wise:
        theta = theta * -1

    theta_arr: ndarray = np.array([0, 0, 0, 0])
    theta_arr[joint_number] = theta

    rot_1_2 = np.matmul(R_1_2, rotation_matrix(theta_arr[0], 'y'))
    rot_2_3 = np.matmul(R_2_3, rotation_matrix(theta_arr[1], 'z'))
    rot_3_4 = np.matmul(R_3_4, rotation_matrix(theta_arr[2], 'z'))
    rot_4_5 = np.matmul(R_4_5, rotation_matrix(theta_arr[3], 'z'))

    rot1_3 = np.matmul(rot_1_2, rot_2_3)
    rot1_4 = np.matmul(rot1_3, rot_3_4)
    rot1_5 = np.matmul(rot1_4, rot_4_5)

    return rot1_5


def print_new_axes(x_arrow, y_arrow, z_arrow, rotate_end_mat):
    # calculate new axis
    new_x_axis = np.matmul(rotate_end_mat, np.array([x_arrow.axis.x, x_arrow.axis.y, x_arrow.axis.z]))
    new_y_axis = np.matmul(rotate_end_mat, np.array([y_arrow.axis.x, y_arrow.axis.y, y_arrow.axis.z]))
    new_z_axis = np.matmul(rotate_end_mat, np.array([z_arrow.axis.x, z_arrow.axis.y, z_arrow.axis.z]))

    x_arrow.axis = vpy.vector(new_x_axis[0], new_x_axis[1], new_x_axis[2])
    y_arrow.axis = vpy.vector(new_y_axis[0], new_y_axis[1], new_y_axis[2])
    z_arrow.axis = vpy.vector(new_z_axis[0], new_z_axis[1], new_z_axis[2])


def update_labels(x_label, y_label, z_label, x_arrow, y_arrow, z_arrow):
    text_offset = 0.025
    x_label.pos = x_arrow.pos + x_arrow.axis + vpy.vector(text_offset, 0, 0)
    y_label.pos = y_arrow.pos + y_arrow.axis + vpy.vector(text_offset, 0, 0)
    z_label.pos = z_arrow.pos + z_arrow.axis + vpy.vector(text_offset, 0, 0)


if __name__ == "__main__":
    offset = 0.025
    rotate_end_matrix = np.ndarray
    THETA = 10

    base_frame_x = vector(1, 0, 0)
    base_frame_y = vector(0, 1, 0)
    base_frame_z = vector(0, 0, 1)

    base_frame_curve_x = vpy.arrow(length=2, pos=origin_vector, axis=base_frame_x, color=color.cyan)
    base_frame_curve_y = vpy.arrow(length=2, pos=origin_vector, axis=base_frame_y, color=color.cyan)
    base_frame_curve_z = vpy.arrow(length=2, pos=origin_vector, axis=base_frame_z, color=color.cyan)

    end_frame_x = vector(1, 0, 0)
    end_frame_y = vector(0, 1, 0)
    end_frame_z = vector(0, 0, 1)

    end_frame_curve_x = vpy.arrow(length=2, pos=origin_vector, axis=end_frame_x, color=color.red)
    end_frame_curve_y = vpy.arrow(length=2, pos=origin_vector, axis=end_frame_y, color=color.red)
    end_frame_curve_z = vpy.arrow(length=2, pos=origin_vector, axis=end_frame_z, color=color.red)

    end_x_label = vpy.label(text="x",
                            color=vpy.color.cyan,
                            pos=end_frame_curve_x.pos + end_frame_curve_x.axis + vpy.vector(offset, 0, 0),
                            box=False)
    end_y_label = vpy.label(text="y",
                            color=vpy.color.cyan,
                            pos=end_frame_curve_y.pos + end_frame_curve_y.axis + vpy.vector(0, offset, 0),
                            box=False)
    end_z_label = vpy.label(text="z", color=vpy.color.cyan,
                            pos=end_frame_curve_z.pos + end_frame_curve_z.axis + vpy.vector(0, 0, offset),
                            box=False)

    base_x_label = vpy.label(text="x",
                             color=vpy.color.red,
                             pos=base_frame_curve_x.pos + base_frame_curve_x.axis + vpy.vector(offset, 0, 0),
                             box=False)
    base_y_label = vpy.label(text="y",
                             color=vpy.color.red,
                             pos=base_frame_curve_y.pos + base_frame_curve_y.axis + vpy.vector(0, offset, 0),
                             box=False)
    base_z_label = vpy.label(text="z",
                             color=vpy.color.red,
                             pos=base_frame_curve_z.pos + base_frame_curve_z.axis + vpy.vector(0, 0, offset),
                             box=False)

    rotate_end_matrix: ndarray = rotate(0, 0)
    print_new_axes(end_frame_curve_x, end_frame_curve_y, end_frame_curve_z, rotate_end_matrix)
    update_labels(end_x_label, end_y_label, end_z_label, end_frame_curve_x, end_frame_curve_y,
                  end_frame_curve_z)

    while True:
        rate(5)
        try:
            if keyboard.is_pressed('z'):
                print('Rotating joint 1 10 degrees counter clock-wise')
                rotate_end_matrix = rotate(THETA, 0)
                print_new_axes(end_frame_curve_x, end_frame_curve_y, end_frame_curve_z, rotate_end_matrix)
                update_labels(end_x_label, end_y_label, end_z_label, end_frame_curve_x, end_frame_curve_y,
                              end_frame_curve_z)

            elif keyboard.is_pressed('a'):
                print('Rotating joint 1 10 degrees clock-wise')
                rotate_end_matrix = rotate(THETA, 0, clock_wise=True)
                print_new_axes(end_frame_curve_x, end_frame_curve_y, end_frame_curve_z, rotate_end_matrix)
                update_labels(end_x_label, end_y_label, end_z_label, end_frame_curve_x, end_frame_curve_y,
                              end_frame_curve_z)

            elif keyboard.is_pressed('x'):
                print('Rotating joint 2 10 degrees counter clock-wise')
                rotate_end_matrix = rotate(THETA, 1)
                print_new_axes(end_frame_curve_x, end_frame_curve_y, end_frame_curve_z, rotate_end_matrix)
                update_labels(end_x_label, end_y_label, end_z_label, end_frame_curve_x, end_frame_curve_y,
                              end_frame_curve_z)

            elif keyboard.is_pressed('s'):
                print('Rotating joint 2 10 degrees clock-wise')
                rotate_end_matrix = rotate(THETA, 1, clock_wise=True)
                print_new_axes(end_frame_curve_x, end_frame_curve_y, end_frame_curve_z, rotate_end_matrix)
                update_labels(end_x_label, end_y_label, end_z_label, end_frame_curve_x, end_frame_curve_y,
                              end_frame_curve_z)

            elif keyboard.is_pressed('c'):
                print('Rotating joint 3 10 degrees counter clock-wise')
                rotate_end_matrix = rotate(THETA, 2)
                print_new_axes(end_frame_curve_x, end_frame_curve_y, end_frame_curve_z, rotate_end_matrix)
                update_labels(end_x_label, end_y_label, end_z_label, end_frame_curve_x, end_frame_curve_y,
                              end_frame_curve_z)

            elif keyboard.is_pressed('d'):
                print('Rotating joint 3 10 degrees clock-wise')
                rotate_end_matrix = rotate(THETA, 2, clock_wise=True)
                print_new_axes(end_frame_curve_x, end_frame_curve_y, end_frame_curve_z, rotate_end_matrix)
                update_labels(end_x_label, end_y_label, end_z_label, end_frame_curve_x, end_frame_curve_y,
                              end_frame_curve_z)

            elif keyboard.is_pressed('f'):
                print('Rotating joint 4 10 degrees counter clock-wise')
                rotate_end_matrix = rotate(THETA, 3)
                print_new_axes(end_frame_curve_x, end_frame_curve_y, end_frame_curve_z, rotate_end_matrix)
                update_labels(end_x_label, end_y_label, end_z_label, end_frame_curve_x, end_frame_curve_y,
                              end_frame_curve_z)

            elif keyboard.is_pressed('v'):
                print('Rotating joint 4 10 degrees clock-wise')
                rotate_end_matrix = rotate(THETA, 3, clock_wise=True)
                print_new_axes(end_frame_curve_x, end_frame_curve_y, end_frame_curve_z, rotate_end_matrix)
                update_labels(end_x_label, end_y_label, end_z_label, end_frame_curve_x, end_frame_curve_y,
                              end_frame_curve_z)

        except Exception as e:
            print(e.with_traceback())
