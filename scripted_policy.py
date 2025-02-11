import json

import numpy as np
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS, PUPPET_JOINT2POS
from ee_sim_env import make_ee_sim_env
from time import sleep
import IPython
e = IPython.embed

import matplotlib

matplotlib.use("QtAgg")
print(matplotlib.get_backend())

from matplotlib import pyplot as plt

def get_policy(task_name, inject_noise=False, obj_names=None):

    if 'sim_transfer_cube' in task_name:
        return PickAndTransferPolicy(inject_noise)
    elif 'sim_insertion' in task_name:
        return InsertionPolicy(inject_noise)
    elif 'general_task' in task_name:
        return GeneralTaskPolicy(inject_noise)
    elif 'primitive' in task_name:
        return PrimitivesPolicy(inject_noise, obj_names)
    else:
        raise NotImplementedError
    
class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.0, 0.1])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]


class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

        ]

class GeneralTaskPolicy(BasePolicy):
    def __init__(self, annotations_file=None, inject_noise=False):
        super().__init__(inject_noise)
        if not annotations_file:
            self.annotations_file = "annotations.json"

    def generate_trajectory(self, ts_first):
        """
        Example annotations file:
            "1": {
            "left": {
                "actions": [
                    "IDLE"
                ],
                "params": {},
                "grabbed_object": null
            },
            "right": {
                "actions": [
                    "IDLE"
                ],
                "params": {},
                "grabbed_object": null
            }
        },
        """
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        gripper_right_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_right_quat = gripper_right_quat * Quaternion(axis=[0.0, 0.0, 1.0], degrees=60)

        gripper_left_quat = Quaternion(init_mocap_pose_left[3:])
        gripper_left_quat = gripper_left_quat * Quaternion(axis=[0.0, 0.0, 1.0], degrees=-60)

        with open(self.annotations_file, 'r') as f:
            annotations = json.load(f)
        
        self.left_trajectory = [{"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}]
        self.right_trajectory = [{"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}]
        
        last_move_position_left  = init_mocap_pose_left[:3]
        last_move_position_right = init_mocap_pose_right[:3]
        last_tilt_quat_left = gripper_left_quat.elements
        last_tilt_quat_right = gripper_right_quat.elements
        last_gripper_left = 1
        last_gripper_right = 1

        new_t = 1
        for key, value in annotations.items():
            t = int(key)
            if t < 500:
                continue
            # Skip if "IDLE" action
            if "IDLE" in value["left"]["actions"] or "IDLE" in value["right"]["actions"]:
                continue
            for arm, entry in value.items():
                if "MOVE" in entry["actions"]:
                    xyz = np.array(entry["params"]["move_position"], dtype='float64')[0]
                    #xyz = np.mean(hand_data, axis=0)
                    if arm == "left":
                        last_move_position_left = xyz
                    else:
                        last_move_position_right = xyz
                else:
                    if arm == "left":
                        xyz = last_move_position_left
                    else:
                        xyz = last_move_position_right
                
                if "TILT" in entry["actions"]:
                    quat = np.array(entry["params"]["tilt_quat"], dtype='float64')[0]
                    if arm == "left":
                        last_tilt_quat_left = quat
                    else:
                        gripper_right_quat = Quaternion(quat)
                        quat = (gripper_right_quat * Quaternion(axis=[0.0, 0.0, 1.0], degrees=60)).elements
                        last_tilt_quat_right = quat
                else:
                    if arm == "left":
                        quat = last_tilt_quat_left
                    else:

                        quat = last_tilt_quat_right

                if "GRAB" in entry["actions"]:
                    gripper = 0

                    if arm == "left":
                        last_gripper_left = 0
                    else:
                        last_gripper_right = 0
                else:
                    if arm == "left":
                        gripper = last_gripper_left
                    else:
                        gripper = last_gripper_right
                            
                if "RELEASE" in entry["actions"]:
                    gripper = 1
                    if arm == "left":
                        last_gripper_left = 1
                    else:
                        last_gripper_right = 1
            
                waypoint = {"t": new_t*15, "xyz": xyz, "quat": quat, "gripper": gripper}
                
                if arm == "left":
                    self.left_trajectory.append(waypoint)
                else:
                    self.right_trajectory.append(waypoint)
                new_t += 1


class PrimitivesPolicy(BasePolicy):
    def __init__(self, inject_noise=False, obj_names=None):
        super().__init__(inject_noise)
        self.obj_names = obj_names
        self.state = {"left": None, "right": None}  # Track current execution phase for each arm
        self.approach_distance = 0.02
        self.approach_params = {"left": None, "right": None}

    def generate_trajectory(self, ts_first):
        pass
    
    def is_reached(self, current, target, threshold=0.01):
        print(f"{current=}, {target=}, {np.linalg.norm(current - target)=}")
        return np.linalg.norm(current - target) < threshold

    def move_and_tilt(self, ts, arm, params, approaching=False):
        """Move to a target position and orientation."""
        mocap_pose = ts.observation[f'mocap_pose_{arm}']
        current_xyz, current_quat = mocap_pose[:3], mocap_pose[3:]
        target_xyz, target_quat = params.get("move_position", None), params.get("tilt_quat", None)
        if not approaching:
            target_xyz = np.array(target_xyz, dtype='float64')[0] if target_xyz is not None else None
            target_quat = np.array(target_quat, dtype='float64')[0] if target_quat is not None else None

        gripper_status = ts.observation["qpos"][6 if arm =="left" else -1]
        # Interpolate movement
        if target_xyz is None:
            new_xyz = current_xyz
            terminated_move = True
        else:
            new_xyz = current_xyz + 0.02 * (target_xyz - current_xyz)
            terminated_move = self.is_reached(current_xyz, target_xyz)
        if target_quat is None:
            terminated_tilt = True
            new_quat = current_quat
        else:
            new_quat = Quaternion.slerp(Quaternion(current_quat), Quaternion(target_quat), 0.1).elements
            terminated_tilt = self.is_reached(current_quat, target_quat)

        return np.concatenate([new_xyz, new_quat, [gripper_status]]), terminated_tilt and terminated_move

    def grab(self, ts, arm, params):
        """Grab an object: move close, then close gripper, then check contact."""
        phase = self.state[arm] or "approach"
        mocap_pose = ts.observation[f'mocap_pose_{arm}']
        current_xyz = mocap_pose[:3]
        target_quat = mocap_pose[3:]
        object_to_grab = params.get("object_grabbed", None)
        if object_to_grab is None:
            if arm == "left":
                object_to_grab = self.obj_names[0]
            else:
                object_to_grab = self.obj_names[-1]

        id_object = self.obj_names.index(object_to_grab)
        
        object_to_grab  = object_to_grab + "_mesh"
        
        if phase == "approach":
            "Get info about the object to grab"
            object_pose = ts.observation["env_state"][id_object*7: id_object*7 + 7]
            params["move_position"] = object_pose[:3]
            #params["tilt_quat"] = object_pose[3:]
            self.approach_params[arm] = params

            action, reached = self.move_and_tilt(ts, arm, params, approaching=True)
            if reached:
                self.state[arm] = "close"
            return action, False
        
        if phase == "close":
            action = np.concatenate([current_xyz, target_quat, [0]])  # Close gripper
            self.state[arm] = "verify"
            return action, False
        
        if phase == "verify":
            gripper_contact = ts.observation[f'contact'][arm]
            if object_to_grab in gripper_contact:
                terminated = True
            else:
                terminated = False

            self.state[arm] = None  # Reset state
            return np.concatenate([current_xyz, target_quat, [0]]), terminated
        
    def release(self, ts, arm):
        """Open gripper to release object."""
        mocap_pose = ts.observation[f'mocap_pose_{arm}']
        action = np.concatenate([mocap_pose[:3], mocap_pose[3:], [1]])  # Open gripper
        return action, True

    def __call__(self, ts, arm, primitive, primitive_params):
        """Execute the requested primitive."""
        print(f"Executing primitive: {primitive} with {arm} arm.")
        if primitive == "MOVE" or primitive == "TILT":
            return self.move_and_tilt(ts, arm, primitive_params)
        if primitive == "GRAB":
            return self.grab(ts, arm, primitive_params)
        if primitive == "RELEASE":
            return self.release(ts, arm)
        
        return np.zeros(8), True

    
def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    elif 'general_task' in task_name:
        env = make_ee_sim_env('general_task')
    elif 'primitive' in task_name:
        env = make_ee_sim_env('primitive')
    else:
        raise NotImplementedError



    # Get mocap positions
    # Get left gripper pose
    #objects=["O02@0094@00001", "O02@0094@00004", "S20005"]

    for episode_idx in range(2):
        ts = env.reset()
        print(ts.observation['contact'])
        exit()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['teleoperator_pov'])
            plt.ion()

        policy = get_policy(task_name, inject_noise, object)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            #print(step, action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['teleoperator_pov'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


def test_policy_primitive(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    
    # setup the environment
    env = make_ee_sim_env('primitive')
    # Get mocap positions
    # Get left gripper pose
    objects=["O02@0094@00001", "O02@0094@00004", "S20005"]
    policy = PrimitivesPolicy(inject_noise, objects)

    with open("annotations.json", 'r') as f:
        annotations = json.load(f)

    annotations_keys = list(annotations.keys())

    ts = env.reset()
    finished = False
    curr_primitive = None
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images']['teleoperator_pov'])
        plt.ion()
    
    p_index = 0
    primitives_to_execute = {"left": [], "right": []}
    while not finished:
        if curr_primitive is None:
            curr_primitive = annotations[annotations_keys[p_index]]
            print(curr_primitive)
            left_params = curr_primitive["left"]["params"]
            right_params = curr_primitive["right"]["params"]

            primitives_to_execute["left"] = curr_primitive["left"]["actions"]
            primitives_to_execute["right"] = curr_primitive["right"]["actions"]

            if "IDLE" in primitives_to_execute["left"]:
                primitives_to_execute["left"] = []
                p_l = None
            else:
                p_l = primitives_to_execute["left"].pop(0)

            if "IDLE" in primitives_to_execute["right"]:
                primitives_to_execute["right"] = []
                p_r = None
            else:
                p_r = primitives_to_execute["right"].pop(0)
            p_index += 1

        finished_primitives = len(primitives_to_execute["left"]) == 0 and len(primitives_to_execute["right"]) == 0
        terminated_left = False
        terminated_right = False

        print(primitives_to_execute, terminated_left, terminated_right, finished_primitives)

        while not finished_primitives:
            obs = ts.observation
            if not terminated_left:
                if p_l is not None:
                    action_left, terminated_left = policy(ts, "left", p_l, left_params)
            else:
                if len(primitives_to_execute["left"]) > 0:
                    p_l = primitives_to_execute["left"].pop(0)
                    action_left, terminated_left = policy(ts, "left", p_l, left_params)
                else:
                    p_l = None
                    # build action left based on the prev obs
                    mocap_pose_left = obs['mocap_pose_left']
                    gripper_status_left = ts.observation["qpos"][6]
                    action_left = np.concatenate([mocap_pose_left[:3], mocap_pose_left[3:], [gripper_status_left]])
                    terminated_left = True

            if not terminated_right:
                if p_r is not None:
                    action_right, terminated_right = policy(ts, "right", p_r, right_params)
            else:
                if len(primitives_to_execute["right"]) > 0:
                    p_r = primitives_to_execute["right"].pop(0)
                    action_right, terminated_right = policy(ts, "right", p_r, right_params)
                else:
                    p_r = None
                    # build action right based on the prev obs
                    mocap_pose_right = obs['mocap_pose_right']
                    gripper_status_right = ts.observation["qpos"][-1]
                    action_right = np.concatenate([mocap_pose_right[:3], mocap_pose_right[3:], [gripper_status_right]])
                    terminated_right = True

            ts = env.step(np.concatenate([action_left, action_right]))

            if onscreen_render:
                plt_img.set_data(ts.observation['images']['teleoperator_pov'])
                plt.pause(0.02)

            finished_primitives = (len(primitives_to_execute["left"]) == 0 and len(primitives_to_execute["right"]) == 0) and (terminated_left and terminated_right)
        
        curr_primitive = None
    plt.close()


if __name__ == '__main__':

    test_task_name = 'primitive'
    #test_policy(test_task_name)
    test_policy_primitive(test_task_name)

