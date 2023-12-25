import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC

class WalkingRobotEnv(gym.Env):
    def __init__(self):
        super(WalkingRobotEnv, self).__init__()
        self.prev_linear_velocity = np.zeros(3)
        self.time_step = 1.0 / 240.0
        # Define action space (10 servos)
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        
        # Define observation space (modify this according to your robot)
        self.observation_space = spaces.Box(
            low=-np.inf,  # Ideally, replace -np.inf with realistic lower bounds
            high=np.inf,  # Similarly, replace np.inf with realistic upper bounds
            shape=(26,),  # Total number of values in the observation vector
            dtype=np.float32
        )

        # Initialize PyBullet
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # Load the robot
        self.robotId = p.loadURDF("C:\\Users\\julia\\codestuff\\hushhopper\\my-robot\\robot.urdf", [0, 0, 0.22], 
                                  p.getQuaternionFromEuler([0, 0, 0]))

    def step(self, action):
        # Apply action to each servo
        for i in range(10):
            p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=action[i])

        # Step the simulation
        p.stepSimulation()

        # Calculate observation, reward, done, info
        observation = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = {}

        return observation, reward, done, info
    
    def reset(self, seed=None, options=None, **kwargs):
        # Reset the robot to a starting state
        p.resetSimulation()
        p.setTimeStep(self.time_step)
        self.robotId = p.loadURDF("C:\\Users\\julia\\codestuff\\hushhopper\\my-robot\\robot.urdf", [0, 0, 0.22], 
                                p.getQuaternionFromEuler([0, 0, 0]))
        self.prev_linear_velocity = [0, 0, 0]  # Resetting the previous linear velocity
        # Additional resets if needed
        initial_observation = self._get_observation()
        return initial_observation

    def render(self, mode='human'):
        # Render the environment (if needed)
        pass

    def close(self):
        # Clean up PyBullet resources
        p.disconnect()

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
        print(observation.shape)
        return observation


    def _compute_reward(self):
        # Example: Reward based on forward movement
        current_position = p.getBasePositionAndOrientation(self.robotId)[0]
        # Assuming forward direction is along the x-axis
        forward_distance = current_position[0]  # You might want to store the initial position to calculate the difference
        reward = forward_distance  # Simple forward movement reward

        # Add other reward components (penalties, goal achievement, etc.)
        return reward


    def _check_done(self):
        # Thresholds
        fall_threshold = 0.2  # Example height threshold indicating a fall
        goal_distance = 10.0  # Example goal distance

        # Check if the robot has fallen
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robotId)
        base_height = base_position[2]
        robot_has_fallen = base_height < fall_threshold

        # Check if the goal is reached
        forward_distance = base_position[0]  # Assuming forward direction is along the x-axis
        reached_goal = forward_distance >= goal_distance

        return robot_has_fallen or reached_goal



# Create the environment
env = WalkingRobotEnv()

# Create the agent
model = SAC("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the agent
model.save("walking_robot_model")
