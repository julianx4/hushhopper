import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
from gymnasium import spaces

ROBOT_URDF_PATH = "my-robot/robot.urdf"

class WalkingRobotEnv(gym.Env):
    def __init__(self, GUI = False):
        super(WalkingRobotEnv, self).__init__()
        self.prev_linear_velocity = np.zeros(3)
        self.time_step = 1.0 / 240.0
        self.step_counter = 0
        # Define action space (10 servos)
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        
        # Define observation space (modify this according to your robot)
        self.observation_space = spaces.Box(
            low=-5,  # Ideally, replace -np.inf with realistic lower bounds
            high=5,  # Similarly, replace np.inf with realistic upper bounds
            shape=(26,),  # Total number of values in the observation vector
            dtype=np.float32
        )

        if GUI == False:
            self.physicsClient = p.connect(p.DIRECT)
        else:
            self.physicsClient = p.connect(p.GUI)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robotId = p.loadURDF(ROBOT_URDF_PATH, [0, 0, 0.22], 
                                  p.getQuaternionFromEuler([0, 0, 0]))
        self.joint_name_to_index = {}
        for i in range(p.getNumJoints(self.robotId)):
            joint_info = p.getJointInfo(self.robotId, i)
            joint_name = joint_info[1].decode("UTF-8")
            self.joint_name_to_index[joint_name] = i  # Map the joint name to its index


    def step(self, action):
        max_velocity = 0.873  # Velocity limit in radians per second
        # Apply action to each servo
        for i in range(10):
            p.setJointMotorControl2(
                bodyUniqueId=self.robotId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=action[i],
                maxVelocity=max_velocity
            )

        # Step the simulation
        p.stepSimulation()
        self.step_counter += 1
        # Calculate observation, reward, done, info
        observation = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = {}
        truncated = False  # If your environment doesn't use truncation
        return observation, reward, done, truncated, info
        
    def reset(self, seed=None, options=None, **kwargs):
        p.resetSimulation()
        p.setTimeStep(self.time_step)

        self._load_robot_and_set_environment()

        self.prev_linear_velocity = [0, 0, 0]
        initial_observation = self._get_observation()
        reset_info = {}
        return initial_observation, reset_info

    def render(self, mode='human'):
        # Render the environment (if needed)
        pass

    def close(self):
        # Clean up PyBullet resources
        p.disconnect()

    def _load_robot_and_set_environment(self):
        p.setGravity(0, 0, -10)
        planeId = p.loadURDF("plane.urdf")
        # plane_friction = 1000.0  # Adjust this value as needed

        # p.changeDynamics(planeId, linkIndex=-1, lateralFriction=plane_friction)

        self.robotId = p.loadURDF(ROBOT_URDF_PATH, [0, 0, 0.22], 
                                  p.getQuaternionFromEuler([0, 0, 0]))
        # foot_friction = 5.0
        # right_foot_index = -1
        # left_foot_index = -1

        # for i in range(p.getNumJoints(self.robotId)):
        #     joint_info = p.getJointInfo(self.robotId, i)
        #     joint_name = joint_info[1].decode("UTF-8")
        #     if joint_name == "right_foot":
        #         right_foot_index = i
        #         p.changeDynamics(self.robotId, right_foot_index, lateralFriction=foot_friction)
        #     elif joint_name == "left_foot":
        #         left_foot_index = i
        #         p.changeDynamics(self.robotId, left_foot_index, lateralFriction=foot_friction)


        initial_base_position = [0, 0, 0.23]
        initial_base_orientation = p.getQuaternionFromEuler([0, 0, 0])  
        p.resetBasePositionAndOrientation(self.robotId, initial_base_position, initial_base_orientation)

        joint_initial_positions = {
            "Left_Foot": 0.7,
            "Left_Hip": -0.7,
            "Left_Hip_Around_Z": 0,
            "Left_Hip_In_Out": 0,
            "Left_Knee": -1.4,
            "Right_Foot": -0.7,
            "Right_Hip": 0.7,
            "Right_Hip_Around_Z": 0,
            "Right_Hip_In_Out": 0,
            "Right_Knee": 1.4
        }

        for joint_name, initial_position in joint_initial_positions.items():
            joint_index = -1
            for i in range(p.getNumJoints(self.robotId)):
                info = p.getJointInfo(self.robotId, i)
                if info[1].decode('UTF-8') == joint_name:
                    joint_index = i
                    break
            if joint_index != -1:
                p.resetJointState(self.robotId, joint_index, initial_position)
            

    def _get_observation(self):
        # Get joint states (positions and velocities for 10 servos)
        joint_states = p.getJointStates(self.robotId, range(10))
        joint_positions = np.array([state[0] for state in joint_states])  # Joint positions
        joint_velocities = np.array([state[1] for state in joint_states])  # Joint velocities

        # Simulate gyroscopic sensor (angular velocity)
        _, angular_velocity = p.getBaseVelocity(self.robotId)
        angular_velocity = np.array(angular_velocity)

        # Simulate accelerometer (linear acceleration)
        current_linear_velocity = np.array(p.getBaseVelocity(self.robotId)[0])
        linear_acceleration = (current_linear_velocity - self.prev_linear_velocity) / self.time_step
        self.prev_linear_velocity = current_linear_velocity

        # Combine all observations into a single array
        observation = np.concatenate([joint_positions, joint_velocities, angular_velocity, linear_acceleration])

        return observation


    def _compute_reward(self):
        current_position, current_orientation = p.getBasePositionAndOrientation(self.robotId)

        forward_distance = 0 #current_position[1]*10.0

        x_axis_rotation, y_axis_rotation, z_axis_rotation = p.getEulerFromQuaternion(current_orientation)
        stability_penalty = 0

        stability_penalty += abs(x_axis_rotation) * 10.0
        stability_penalty += abs(y_axis_rotation) * 10.0
        stability_penalty += abs(z_axis_rotation) * 10.0

        base_height = current_position[2]
        height_penalty = abs(base_height - 0.22) * 100.0
        
        # joint_penalty = 0
        # penalty_scaling_factor = 10.0  # Adjust this to scale the joint penalty
        # specific_joints = ["Left_Hip_Around_Z", "Left_Hip_In_Out", 
        #                 "Right_Hip_Around_Z", "Right_Hip_In_Out"]

        # for joint_name in specific_joints:
        #     joint_index = self.joint_name_to_index[joint_name]
        #     joint_state = p.getJointState(self.robotId, joint_index)
        #     joint_position = joint_state[0]  # Get the position of the joint
        #     joint_penalty += abs(joint_position)  # Penalize based on the absolute deviation from 0

        # joint_penalty *= penalty_scaling_factor
        time_reward = self.step_counter * 0.01
        reward = time_reward
        return reward

    def _check_done(self):
        # Thresholds
        fall_threshold = 0.15  # Example height threshold indicating a fall
        goal_distance = 10.0  # Example goal distance

        # Check if the robot has fallen
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robotId)
        base_height = base_position[2]
        robot_has_fallen = base_height < fall_threshold

        # Check if the goal is reached
        forward_distance = base_position[1]  # Assuming forward direction is along the x-axis
        reached_goal = forward_distance >= goal_distance

        return robot_has_fallen or reached_goal