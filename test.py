import vpython as vpy

motor_1: vpy.cylinder = vpy.cylinder(pos=vpy.vector(0, 0, 0), radius=0.3, axis=vpy.vector(0, 0, 1));
rod1_1: vpy.cylinder = vpy.cylinder(pos=vpy.vector(0, 0, 1), radius=0.2, axis=vpy.vector(0, 0, 1.5))
part_1: vpy.compound = vpy.compound([motor_1, rod1_1])
part_1.color = vpy.color.red

motor_2: vpy.cylinder = vpy.cylinder(pos=vpy.vector(0, -1.5, 0), radius=0.3, axis=vpy.vector(0, 3, 0))
rod2_1: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x, -motor_2.pos.y, motor_2.pos.z), radius=0.2,
                                    axis=vpy.vector(3, 0, 0))
rod2_2: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x, motor_2.pos.y, motor_2.pos.z), radius=0.2,
                                    axis=vpy.vector(3, 0, 0))
rod2_3: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + 3, motor_2.pos.y, motor_2.pos.z), radius=0.2,
                                    axis=vpy.vector(0, 3, 0))
rod2_4: vpy.cylinder = vpy.cylinder(pos=vpy.vector(motor_2.pos.x + 3, motor_2.pos.y + 1.5, motor_2.pos.z), radius=0.2,
                                    axis=vpy.vector(3, 0, 0))
part_2: vpy.compound = vpy.compound([motor_2, rod2_4, rod2_3, rod2_1, rod2_2], pos=vpy.vector(3, 0, 2.5))
part_2.color = vpy.color.blue

part_3: vpy.compound = part_2.clone(pos=vpy.vector(part_2.pos.x + 6, part_2.pos.y, part_2.pos.z))
part_3.color = vpy.color.green
# part_3.rotate(angle=vpy.radians(30), axis=vpy.vector(0, 1, 0))
part_2.rotate(angle=vpy.radians(-30), axis=vpy.vector(0, 1, 0), origin=vpy.vector(0, 0, 2.5))