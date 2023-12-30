import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,0.4]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("mini_cheetah/mini_cheetah.urdf",cubeStartPos, cubeStartOrientation, 
                   # useMaximalCoordinates=1, ## New feature in Pybullet
                   flags=p.URDF_USE_INERTIA_FROM_FILE)
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
    current_position, current_orientation = p.getBasePositionAndOrientation(robotId)
    x_axis_rotation, y_axis_rotation, z_axis_rotation = p.getEulerFromQuaternion(current_orientation)
    if abs(x_axis_rotation) > 1.5 or abs(y_axis_rotation) > 1.5:
        print("Robot fell over!")
    print(current_position[2])
cubePos, cubeOrn = p.getBasePositionAndOrientation(robotId)
print(cubePos,cubeOrn)
p.disconnect()

