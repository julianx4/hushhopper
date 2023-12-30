import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete
np.set_printoptions(precision=3, linewidth=np.inf)

ROBOT_URDF_PATH = "my-robot/robot.urdf"

class WalkingRobotEnv(gym.Env):
    def __init__(self, GUI = False):
        super(WalkingRobotEnv, self).__init__()
        self.prev_linear_velocity = np.zeros(3)
        self.time_step = 1.0 / 120.0
        self.step_counter = 0
        self.episode_step_counter = 0

        self.action_space = MultiDiscrete([7] * 10)
        
        # Define observation space (modify this according to your robot)
        self.observation_space = spaces.Box(
            low=-50,
            high=50,  
            shape=(26,),  
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

    def step(self, action):
        max_velocity = np.pi / 3  # Adjusted to real servo speed

        angle_changes = self.scale_actions(action)

        # Apply the angle changes to each joint independently
        for i in range(len(angle_changes)):  # Assuming the robot has as many joints as the length of angle_changes
            current_position = p.getJointState(self.robotId, i)[0]
            new_position = current_position + angle_changes[i]
            p.setJointMotorControl2(
                bodyUniqueId=self.robotId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=new_position,
                maxVelocity=max_velocity  # Set an appropriate max velocity
            )
        #print(angle_changes)
        # Step the simulation
        p.stepSimulation()
        self.step_counter += 1
        self.episode_step_counter += 1
        observation = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = {}
        truncated = False
        return observation, reward, done, truncated, info
        
    def reset(self, seed=None, options=None, **kwargs):
        p.resetSimulation()
        p.setTimeStep(self.time_step)

        self._load_robot_and_set_environment()
        self.episode_step_counter = 0

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

    def get_joint_limits(self):
        num_joints = p.getNumJoints(self.robotId)
        lower_limits = np.zeros(num_joints)
        upper_limits = np.zeros(num_joints)

        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(self.robotId, joint_index)
            lower_limits[joint_index], upper_limits[joint_index] = joint_info[8], joint_info[9]
        print("lower limits: ", lower_limits, "upper limits: ", upper_limits)
        return lower_limits, upper_limits

    def scale_actions(self, actions):
        angle_changes = np.zeros(len(actions))
        for i, action in enumerate(actions):
            if action == 0:
                angle_changes[i] = -np.deg2rad(20)  # -20 degrees
            elif action == 1:
                angle_changes[i] = -np.deg2rad(5)  # -5 degrees
            elif action == 2:
                angle_changes[i] = -np.deg2rad(1)  # -1 degree
            elif action == 3:
                angle_changes[i] = 0  # No movement
            elif action == 4:
                angle_changes[i] = np.deg2rad(1)  # 1 degree
            elif action == 5:
                angle_changes[i] = np.deg2rad(5)  # 5 degrees
            elif action == 6:
                angle_changes[i] = np.deg2rad(20)  # 20 degrees

        return angle_changes


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


    def _get_observation(self):
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
        observation = np.concatenate([joint_positions, joint_velocities, angular_velocity, linear_acceleration]).astype(np.float32)

        return observation

    def _compute_reward(self):
        current_position, current_orientation = p.getBasePositionAndOrientation(self.robotId)

        base_height = current_position[2]
        height_penalty = abs(base_height - 0.2378)

        # Penalty if deviation from original joint position:
        deviation_penalty = 0
        joint_states = p.getJointStates(self.robotId, range(p.getNumJoints(self.robotId)))

        # Iterate over the joint names and their initial positions
        for joint_name, initial_position in self.joint_initial_positions.items():
            joint_index = self.joint_name_to_index[joint_name]  # Get the index of the joint
            joint_position = joint_states[joint_index][0]  # Current joint position
            deviation_penalty += abs(joint_position - initial_position)

            
        deviation_penalty_weight = 0.1
        height_penalty_weight = 1.0  # Adjust based on importance of height maintenance
        survival_reward = self.episode_step_counter * 0.01  # Reward for staying upright

        reward = -height_penalty * height_penalty_weight - deviation_penalty * deviation_penalty_weight + survival_reward
        return reward


    def _compute_reward2(self):
        current_position, current_orientation = p.getBasePositionAndOrientation(self.robotId)

        forward_distance = current_position[1]*100.0
        side_distance = abs(current_position[0])*100.0

        x_axis_rotation, y_axis_rotation, z_axis_rotation = p.getEulerFromQuaternion(current_orientation)
        stability_penalty = 0

        stability_penalty += abs(z_axis_rotation) * 10.0

        base_height = current_position[2]
        height_penalty = abs(base_height - 0.2) * 30.0
        #print("baseheight",base_height)
        
        time_penalty = 0 # self.episode_step_counter * 0.01
        #print("time penalty: ",time_penalty, "forward distance: ", forward_distance)
        reward = forward_distance - side_distance - time_penalty - stability_penalty # - joint_penalty
        return reward

    def _check_done(self):
        # Thresholds
        goal_distance = 10.0  # Example goal distance
        wrong_direction_limit = 1.0  # How far off-center the robot can be before terminating the episode

        # Check if the robot has fallen
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robotId)
        x_axis_rotation, y_axis_rotation, z_axis_rotation = p.getEulerFromQuaternion(base_orientation)
        robot_has_fallen = False
        if abs(x_axis_rotation) > 1.5 or abs(y_axis_rotation) > 1.5:
            robot_has_fallen = True

        robot_turned_too_much = False
        if abs(z_axis_rotation) > 0.3: #0.3 rad in degrees is 17.18 degrees
            robot_turned_too_much = True

        # Check if the goal is reached
        forward_distance = base_position[1]
        side_distance = abs(base_position[0])

        reached_goal = forward_distance >= goal_distance
        strayed_off_course = side_distance >= wrong_direction_limit
        walked_backward_too_far = forward_distance < -wrong_direction_limit
        return robot_has_fallen # or reached_goal or strayed_off_course or walked_backward_too_far or robot_turned_too_much