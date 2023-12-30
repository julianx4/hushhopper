import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-0.005)
planeId = p.loadURDF("./plane.urdf")

p.changeDynamics(planeId, -1, lateralFriction=1,
            spinningFriction=0, rollingFriction=0)
cubeStartPos = [0,0,0.22]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("my-robot/robot.urdf",cubeStartPos, cubeStartOrientation)
joint_initial_positions = {
    "left_foot": -0.7,
    "left_hip": -0.7,
    "left_hip_around_torso": 0,
    "left_in_out": 0,
    "left_knee": 1.4,
    "right_foot": 0.7,
    "right_hip": 0.7,
    "right_hip_around_torso": 0,
    "right_in_out": 0,
    "right_knee": -1.4
}

for joint_name, initial_position in joint_initial_positions.items():
    joint_index = -1
    for i in range(p.getNumJoints(robotId)):
        info = p.getJointInfo(robotId, i)
        if info[1].decode('UTF-8') == joint_name:
            joint_index = i
            break
    if joint_index != -1:
        p.resetJointState(robotId, joint_index, initial_position)

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
    current_position, current_orientation = p.getBasePositionAndOrientation(robotId)
    print(current_position)
cubePos, cubeOrn = p.getBasePositionAndOrientation(robotId)
print(cubePos,cubeOrn)
p.disconnect()

