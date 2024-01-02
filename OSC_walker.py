import pybullet as p
import pybullet_data
import numpy as np
import time
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
import threading

Z_POSITION = 0.24
EARTH_GRAVITY = 9.81
MOON_GRAVITY = 1.62

np.set_printoptions(precision=3, linewidth=np.inf)

ROBOT_URDF_PATH = "hushhopper3D/robot.urdf"

class Robot():
    def __init__(self):
        self.prev_linear_velocity = np.zeros(3)
        self.time_step = 1.0 / 240.0
        self.step_counter = 0
        self.episode_step_counter = 0
        self.physicsClient = p.connect(p.GUI)
        self.gravity = -EARTH_GRAVITY
        
        p.setRealTimeSimulation(1)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robotId = p.loadURDF(ROBOT_URDF_PATH, [0, 0, Z_POSITION], 
                                  p.getQuaternionFromEuler([0, 0, 0]))
        self.joint_name_to_index = {}
        for i in range(p.getNumJoints(self.robotId)):
            joint_info = p.getJointInfo(self.robotId, i)
            joint_name = joint_info[1].decode("UTF-8")
            self.joint_name_to_index[joint_name] = i  # Map the joint name to its index
        self.lower_limits, self.upper_limits = self.get_joint_limits()


        self.joint_order = [
            "left_foot", "left_hip", "left_Z", "left_in_out", 
            "left_knee", "right_foot", "right_hip", "right_Z", 
            "right_in_out", "right_knee"
        ]
        self.joint_initial_positions = {
            "left_foot": -0.7,
            "left_hip": -1,
            "left_Z": 0,
            "left_in_out": 0,
            "left_knee": 1.7,
            "right_foot": 0.7,
            "right_hip": 1,
            "right_Z": 0,
            "right_in_out": 0,
            "right_knee": -1.7
        }
        self.new_foot_1_position_rel = (0, 0, 0)
        self.new_foot_2_position_rel = (0, 0, 0)
        self.link_name_to_index_map = self.create_link_name_to_index_map()

        self.joint_angles = self.joint_initial_positions.copy()
        self.max_velocity = 0.8

        self.keep_running = True
        self.lock = threading.Lock()
        self.joint_angles_for_pybullet = {}
        
    def create_link_name_to_index_map(self):
        link_name_to_index = {}
        for i in range(p.getNumJoints(self.robotId)):
            link_info = p.getJointInfo(self.robotId, i)
            link_name = link_info[12].decode('UTF-8')
            link_name_to_index[link_name] = i
        return link_name_to_index
    
    def get_link_position_and_orientation_by_name(self, link_name):
        if link_name in self.link_name_to_index_map:
            link_index = self.link_name_to_index_map[link_name]
            link_state = p.getLinkState(self.robotId, link_index)
            link_position = link_state[0]
            link_orientation = link_state[1]
            return link_position, link_orientation
        else:
            raise ValueError(f"Link name '{link_name}' not found in the robot.")
        
    def transform_position_to_world_frame(self, position_rel, base_pos, base_ori):
        rotated_pos = p.rotateVector(base_ori, position_rel)
        world_pos = [rotated_pos[0] + base_pos[0], rotated_pos[1] + base_pos[1], rotated_pos[2] + base_pos[2]]
        return world_pos

    def transform_orientation_to_world_frame(self, orientation_rel, base_ori):
        world_ori = p.multiplyTransforms([0, 0, 0], base_ori, [0, 0, 0], orientation_rel)[1]
        return world_ori
           
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
        p.setGravity(0, 0, self.gravity)
        planeId = p.loadURDF("./plane.urdf")

        p.changeDynamics(planeId, -1, lateralFriction=1,
                    spinningFriction=0, rollingFriction=0)

        self.robotId = p.loadURDF(ROBOT_URDF_PATH, [0, 0, Z_POSITION], 
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
            
        initial_base_position = [0, 0, Z_POSITION]
        initial_base_orientation = p.getQuaternionFromEuler([0, 0, 0])  
        p.resetBasePositionAndOrientation(self.robotId, initial_base_position, initial_base_orientation)

    def start_osc_server(self, ip, port):
        dispatcher = Dispatcher()
        dispatcher.map("/update_joints", self.handle_joint_updates)

        self.osc_server = ThreadingOSCUDPServer((ip, port), dispatcher)
        print(f"Starting OSC server on {ip}:{port}")
        server_thread = threading.Thread(target=self.osc_server.serve_forever)
        server_thread.start()

    def handle_joint_updates(self, *angles):
        angles = angles[1:]
        #print("Received joint angles: ", angles)
        if len(angles) != len(self.joint_order):
            print("Error: Received incorrect number of joint angles")
            return
        with self.lock:
            self.joint_angles_for_pybullet  = dict(zip(self.joint_order, angles))

    def run_simulation(self):
        while self.keep_running:

            keys = p.getKeyboardEvents()
        
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                self.reset()

            if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
                self.gravity = 0 if self.gravity == -EARTH_GRAVITY else -EARTH_GRAVITY
                print("Gravity: ", self.gravity)
                p.setGravity(0, 0, self.gravity)

            with self.lock:
                for joint_name, angle in self.joint_angles_for_pybullet.items():
                    joint_index = self.joint_name_to_index.get(joint_name)
                    if joint_index is not None:
                        p.setJointMotorControl2(self.robotId, joint_index, p.POSITION_CONTROL, 
                                                targetPosition=angle, force=500, maxVelocity=self.max_velocity)
                    else:
                        print(f"Joint {joint_name} not found")

            p.stepSimulation()
            time.sleep(1.0 / 240.0)  # Simulation time step

    def stop(self):
        self.keep_running = False
        #self.osc_server.shutdown()  # Stop the OSC server

# Example usage
robot = Robot()
robot.reset()
robot.start_osc_server("192.168.2.107", 9005)

try:
    robot.run_simulation()

except KeyboardInterrupt:
    print("Stopping simulation")
    robot.stop()