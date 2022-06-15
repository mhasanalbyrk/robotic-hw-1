import numpy as np
import vpython as vpy

global H01
H12 = None
H23 = None
H34 = None
H45 = None
H56 = None

a1 = 10
a2 = 10
a3 = 10
a4 = 1
a5 = 1
a6 = 1

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


def rotation_matrix(degrees, axis='z'):
    theta_degrees = np.radians(degrees)

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
                    [a1]])
    d12 = np.array([[a2 * np.cos(theta2)],
                    [a2 * np.sin(theta2)],
                    [0]])
    d23 = np.array([[a3 * np.cos(theta3)],
                    [a3 * np.sin(theta3)],
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
                                                                 [1, 3])).reshape([3,1])])
    jacobianUpper = np.column_stack([jacobianUpper, np.cross((H02[0:3, 0:3] @ np.array([[0],
                                                                                        [0],
                                                                                        [1]]).flatten()),
                                                             H06[0:3, 3].reshape([1, 3]) - H02[0:3, 3].reshape(
                                                                 [1, 3])).reshape([3,1])])
    jacobianUpper = np.column_stack([jacobianUpper, np.cross(H03[0:3, 0:3] @ np.array([[0],
                                                                                       [0],
                                                                                       [1]]).flatten(),
                                                             H06[0:3, 3].reshape([1, 3]) - H03[0:3, 3].reshape(
                                                                 [1, 3])).reshape([3,1])])
    jacobianUpper = np.column_stack([jacobianUpper, np.cross((H04[0:3, 0:3] @ np.array([[0],
                                                                                        [0],
                                                                                        [1]]).flatten()),
                                                             H06[0:3, 3].reshape([1, 3]) - H04[0:3, 3].reshape(
                                                                 [1, 3])).reshape([3,1])])
    jacobianUpper = np.column_stack([jacobianUpper, np.cross((H05[0:3, 0:3] @ np.array([[0],
                                                                                        [0],
                                                                                        [1]]).flatten()),
                                                             H06[0:3, 3].reshape([1, 3]) - H05[0:3, 3].reshape(
                                                                 [1, 3])).reshape([3,1])])

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
    xDot = 0
    yDot = 5
    zDot = 0
    wX = 0
    wY = 0
    wZ = 0
    theta1 = 0
    theta2 = 0
    theta3 = 0
    theta4 = 0
    theta5 = 90
    theta6 = 0
    offset = 0.0
    THETA_UNIT = 5
    motor_1: vpy.cylinder = vpy.cylinder(pos=vpy.vector(0, 0, 0), radius=0.3, axis=vpy.vector(0, 0, 1));
    rod1_1: vpy.cylinder = vpy.cylinder(pos=vpy.vector(0, 0, 1), radius=0.2, axis=vpy.vector(0, 0, a1))
    part_1: vpy.compound = vpy.compound([motor_1, rod1_1])
    part_1.color = vpy.color.red
    original_motor_1_axis = [0, 3, 0]
    original_motor_2_axis = [0, 3, 0]
    motor_2: vpy.cylinder = vpy.cylinder(pos=vpy.vector(0, -1.5, a1), radius=0.3, axis=vpy.vector(0, 3, 0))
    motor_2.color = vpy.color.green
    rod2_1: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x, -motor_2.pos.y, motor_2.pos.z), radius=0.2,
                                        axis=vpy.vector(a2/2, 0, 0))
    rod2_1.color = vpy.color.cyan
    rod2_2: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x, motor_2.pos.y, motor_2.pos.z), radius=0.2,
                                        axis=vpy.vector(a2/2, 0, 0))
    rod2_2.color = vpy.color.magenta
    rod2_3: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + a2/2, motor_2.pos.y, motor_2.pos.z), radius=0.2,
                                        axis=vpy.vector(0, 3, 0))
    rod2_3.color = vpy.color.yellow

    rod2_4: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + a2/2, motor_2.pos.y + 1.5, motor_2.pos.z),
                                        radius=0.2,
                                        axis=vpy.vector(a3/2, 0, 0))

    motor_3: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + a2, motor_2.pos.y + 1.5, a1), radius=0.3,
                                         axis=vpy.vector(0, 3, 0))
    motor_3.color = vpy.color.blue

    rod3_1: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + a2, -motor_2.pos.y, motor_3.pos.z), radius=0.2,
                                        axis=vpy.vector(a3/2, 0, 0))
    rod3_1.color = vpy.color.cyan

    rod3_2: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + a2, motor_2.pos.y, motor_3.pos.z), radius=0.2,
                                        axis=vpy.vector(a3/2, 0, 0))
    rod3_2.color = vpy.color.magenta

    rod3_3: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + a2/2 + a2, motor_2.pos.y, motor_3.pos.z), radius=0.2,
                                        axis=vpy.vector(0, 3, 0))
    rod3_3.color = vpy.color.yellow

    rod3_4: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x +a2/2 + a2, rod2_4.pos.y, motor_3.pos.z),
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

    movVars = np.array([[xDot],
                       [yDot],
                       [zDot],
                       [wX],
                       [wY],
                       [wZ]])

    thetas = np.array([
        [theta1],
        [theta2],
        [theta3],
        [theta4],
        [theta5],
        [theta6]])
    calculateHomoMatricies(thetas[0][0], thetas[1][0], thetas[2][0], thetas[3][0], thetas[4][0], thetas[5][0])
    jacobian = calculateJacobian()

    y = 0
    # while(True):
    #     calculateHomoMatricies(thetas[0][0], thetas[1][0], thetas[2][0], thetas[3][0], thetas[4][0], thetas[5][0])
    #     jacobian = calculateJacobian()
    #     deltas = np.linalg.pinv(jacobian) @ movVars

