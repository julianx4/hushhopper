import pybullet as p
import pybullet_data
import numpy as np
import time

np.set_printoptions(precision=3, linewidth=np.inf)

ROBOT_URDF_PATH = "my-robot/robot.urdf"

class Robot():
    def __init__(self):
        self.prev_linear_velocity = np.zeros(3)
        self.time_step = 1.0 / 480.0
        self.step_counter = 0
        self.episode_step_counter = 0
        self.physicsClient = p.connect(p.GUI)
        p.setRealTimeSimulation(1)


        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robotId = p.loadURDF(ROBOT_URDF_PATH, [0, 0, 0.22], 
                                  p.getQuaternionFromEuler([0, 0, 0]))
        self.joint_name_to_index = {}
        for i in range(p.getNumJoints(self.robotId)):
            joint_info = p.getJointInfo(self.robotId, i)
            joint_name = joint_info[1].decode("UTF-8")
            self.joint_name_to_index[joint_name] = i  # Map the joint name to its index
        self.lower_limits, self.upper_limits = self.get_joint_limits()

        self.joint_initial_positions = {
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
        

        
    def reset(self):
        p.resetSimulation()
        p.setTimeStep(self.time_step)

        self._load_robot_and_set_environment()
        self.episode_step_counter = 0

        self.prev_linear_velocity = [0, 0, 0]
        reset_info = {}
        return 
    
    def render(self, mode='human'):
        # Render the environment (if needed)
        pass

    def close(self):
        # Clean up PyBullet resources
        p.disconnect()

    def get_joint_limits(self):
        num_joints = p.getNumJoints(self.robotId)
        lower_limits = np.zeros(num_joints)
        upper_limits = np.zeros(num_joints)

        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(self.robotId, joint_index)
            lower_limits[joint_index], upper_limits[joint_index] = joint_info[8], joint_info[9]
        print("lower limits: ", lower_limits, "upper limits: ", upper_limits)
        return lower_limits, upper_limits


    def _load_robot_and_set_environment(self):
        p.setGravity(0, 0, -9.81)
        planeId = p.loadURDF("./plane.urdf")

        p.changeDynamics(planeId, -1, lateralFriction=1,
                    spinningFriction=0, rollingFriction=0)

        self.robotId = p.loadURDF(ROBOT_URDF_PATH, [0, 0, 0.22], 
                                  p.getQuaternionFromEuler([0, 0, 0]))



        for joint_name, initial_position in self.joint_initial_positions.items():
            joint_index = -1
            for i in range(p.getNumJoints(self.robotId)):
                info = p.getJointInfo(self.robotId, i)
                if info[1].decode('UTF-8') == joint_name:
                    joint_index = i
                    break
            if joint_index != -1:
                p.resetJointState(self.robotId, joint_index, initial_position)
            
        initial_base_position = [0, 0, 0.22]
        initial_base_orientation = p.getQuaternionFromEuler([0, 0, 0])  
        p.resetBasePositionAndOrientation(self.robotId, initial_base_position, initial_base_orientation)

    def do_squats(self, repetitions=10):
        max_velocity = 0.8
        self.joint_squat_positions = {
            "left_foot": -1,
            "left_hip": -1,
            "left_hip_around_torso": 0,
            "left_in_out": 0,
            "left_knee": 2,
            "right_foot": 1,
            "right_hip": 1,
            "right_hip_around_torso": 0,
            "right_in_out": 0,
            "right_knee": -2
        }
        for i in range(repetitions):
            self._move_to_joint_positions(self.joint_squat_positions, max_velocity)
            self._move_to_joint_positions(self.joint_initial_positions, max_velocity)


    def do_step(self, repetitions=10):
        max_velocity = 0.8
        self.joint_step_position = {
            "right_foot": 0,
            "right_hip": 0,
            "right_hip_around_torso": 0,
            "right_in_out": 0,
            "right_knee": 0
        }

        self._move_to_joint_positions(self.joint_step_position, max_velocity)

    def do_step_setup(self, repetitions=10):
        max_velocity = 0.8
        self.step_setup_positions = {
            "left_foot": -1,
            "left_hip": -0.7,
            "left_hip_around_torso": 0,
            "left_in_out": 0,
            "left_knee": 2,
            "right_foot": 1,
            "right_hip": 0.7,
            "right_hip_around_torso": 0,
            "right_in_out": 0,
            "right_knee": -2
        }

        self._move_to_joint_positions(self.step_setup_positions, max_velocity)

    def _move_to_joint_positions(self, joint_positions, max_velocity):
        # Calculate the current position of each joint
        current_positions = [p.getJointState(self.robotId, self.joint_name_to_index[joint_name])[0] for joint_name in joint_positions.keys()]

        # Calculate distances to the target positions
        distances = [abs(joint_positions[joint_name] - current_positions[i]) for i, joint_name in enumerate(joint_positions.keys())]
        max_distance = max(distances)

        for i, (joint_name, joint_position) in enumerate(joint_positions.items()):
            joint_index = self.joint_name_to_index[joint_name]

            # Calculate velocity for this joint based on its distance to the target
            # Ensuring that the joint with the max distance uses the max_velocity
            velocity = (distances[i] / max_distance) * max_velocity if max_distance > 0 else 0

            # Set joint motor control with the calculated velocity
            p.setJointMotorControl2(self.robotId, joint_index, p.POSITION_CONTROL, 
                                    targetPosition=joint_position, force=500, maxVelocity=velocity)

        for _ in range(480):
            p.stepSimulation()
            time.sleep(0.01)  # Sleep for 10 milliseconds


robot = Robot()
robot.reset()
time.sleep(1)
robot.do_step_setup()
time.sleep(1)
robot.do_step(repetitions=1)