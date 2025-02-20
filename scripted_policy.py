import os

import json
import numpy as np
from pyquaternion import Quaternion
import open3d as o3d
import cv2
from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from policy_utils.policy_utils import *
import PIL
from time import sleep
from matplotlib import pyplot as plt

os.environ["MUJOCO_GL"] = "egl"

# ENVIRONMENT CONFIGS
LEFT_CAM_ID = 2
RIGHT_CAM_ID = 3

# ----------------- POLICIES ----------------- #
    
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


class PrimitivesPolicy(BasePolicy):
    def __init__(self, inject_noise=False, obj_names=None, camera_intr=None, affordance_model=None):
        super().__init__(inject_noise)
        self.obj_names = obj_names
        self.state = {"left": None, "right": None}  # Track current execution phase for each arm
        self.left_approach_distance = [0, 0, 0.05]
        self.right_approach_distance = [0, -0.05, 0.05]
        self.approach_params = {"left": None, "right": None}
        self.affordance_model = affordance_model
        self.camera_intr = camera_intr

    def generate_trajectory(self, ts_first):
        pass
        

    def is_reached(self, current, target, threshold=0.01):
        print(f"{current}, {target}, {np.linalg.norm(current - target)}")
        return np.linalg.norm(current - target) < threshold
    
    def move_and_tilt(self, ts, arm, params, approaching=False, threshold=0.03):
        """Move to a target position and orientation."""
        mocap_pose = ts.observation[f'mocap_pose_{arm}']
        current_xyz, current_quat = mocap_pose[:3], mocap_pose[3:]
        
        target_xyz, target_quat = params.get("move_position", None), params.get("tilt_quat", None)

        #target_xyz, target_quat = params.get("move_position", None), current_quat
        if not approaching:
            target_xyz = np.array(target_xyz, dtype='float64')[0] if target_xyz is not None else None
            target_quat = np.array(target_quat, dtype='float64')[12] if target_quat is not None else None

            if arm == "left" and target_quat is not None:
                lq = Quaternion(target_quat)
                lq = lq * Quaternion(axis=[0.0, 0.0, 1.0], degrees=-60)
                target_quat = lq.elements
            elif arm == "right" and target_quat is not None:
                rq = Quaternion(target_quat)
                rq = rq * Quaternion(axis=[0.0, 0.0, 1.0], degrees=60)
                target_quat = rq.elements
        else:
            # set target quat to gripper towards the table
            target_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-70).elements

        gripper_status = ts.observation["qpos"][6 if arm =="left" else -1]
        # Interpolate movement
        if target_xyz is None:
            new_xyz = current_xyz
            terminated_move = True
        else:
            new_xyz = current_xyz + 0.05 * (target_xyz - current_xyz)
            terminated_move = self.is_reached(current_xyz, target_xyz, threshold)

        if target_quat is None:
            terminated_tilt = True
            new_quat = current_quat
        else:
            new_quat = Quaternion.slerp(Quaternion(current_quat), Quaternion(target_quat), 0.1).elements
            terminated_tilt = self.is_reached(current_quat, target_quat, threshold)

        return np.concatenate([new_xyz, new_quat, [gripper_status]]), terminated_tilt and terminated_move

    def grab(self, ts, arm, params):
        """Grab an object: move close, then close gripper, then check contact."""
        phase = self.state[arm] or "setup"
        mocap_pose = ts.observation[f'mocap_pose_{arm}']
        current_xyz = mocap_pose[:3]
        current_quat = mocap_pose[3:]
        object_to_grab = params.get("object_grabbed", None)
        if object_to_grab is None:
            if arm == "left":
                object_to_grab = self.obj_names[0]
            else:
                object_to_grab = self.obj_names[-1]

        id_object = self.obj_names.index(object_to_grab)
        
        object_to_grab  = object_to_grab + "_mesh"
        print("Phase: ", phase)
        if phase == "setup":
            # "Get info about the object to grab"
            
            object_pose = ts.observation["env_state"][id_object*7: id_object*7 + 7]
            #best_quat = self.compute_grasp_quat(object_pose[3:], "knife")
            """if arm == "left":
                params["move_position"] = object_pose[:3] + np.array(self.left_approach_distance)
                self.left_approach_distance[-1] -= 0.02
            else:
                params["move_position"] = object_pose[:3] + np.array(self.right_approach_distance)
                self.right_approach_distance[-1] -= 0.02

            q_grasp = Quaternion(object_pose[3:]) * Quaternion(axis=[1, 0, 0], angle=3.1416)  # Rotate by π around X

            params["tilt_quat"] = q_grasp.elements
            self.approach_params[arm] = params
            
            action, reached = self.move_and_tilt(ts, arm, params, approaching=True, threshold=0.03)

            if reached and self.approach_distance[-1] <= 0:
                self.state[arm] = "close"
                if arm == "left":
                    self.left_approach_distance = [0, 0, 0.05]
                else:
                    self.right_approach_distance = [0, -0.05, 0.05]"""
            params["move_position"] = object_pose[:3] + np.array([0, -0.1, 0.2])
            q_grasp = Quaternion(object_pose[3:]) * Quaternion(axis=[1, 0, 0], angle=3.1416)  # Rotate by π around X

            params["tilt_quat"] = q_grasp

            action, reached = self.move_and_tilt(ts, arm, params, approaching=True, threshold=0.03)
            if reached:
                self.state[arm] = "affordance"
            return action, False
        
        if phase == "affordance":
            
            # get pointcloud from depth
            rgb = o3d.geometry.Image(ts.observation['images'][f'{arm}_wrist'])
            depth = o3d.geometry.Image(ts.observation['images'][f'{arm}_depth'])
            width, height = rgb.width, rgb.height
            fx, fy, cx, cy = self.camera_intr[arm]
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(
                    width=width,
                    height=height,
                    fx=fx, fy=fy,
                    cx=cx, cy=cy
                )
            )
            pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            
            if reached:
                self.state[arm] = "close"
            return action, False
        
        if phase == "approach":
            
            if reached:
                self.state[arm] = "close"
            return action, False

        if phase == "close":
            action = np.concatenate([current_xyz, current_quat, [0]])  # Close gripper
            self.state[arm] = "verify"
            return action, False
        
        if phase == "verify":
            gripper_contact = ts.observation[f'contact'][arm]
            print(f"{gripper_contact}")
            if object_to_grab in gripper_contact:
                terminated = True
            else:
                terminated = False

            self.state[arm] = None  # Reset state
            return np.concatenate([current_xyz, current_quat, [0]]), terminated
        
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
            """mocap_pose = ts.observation[f'mocap_pose_{arm}']
            current_xyz, current_quat = mocap_pose[:3], mocap_pose[3:]
            gripper_status = ts.observation["qpos"][6 if arm =="left" else -1]
            return np.concatenate([current_xyz, current_quat, [gripper_status]]), True"""
            return self.grab(ts, arm, primitive_params)
        if primitive == "RELEASE":
            return self.release(ts, arm)
        
        return np.zeros(8), True

    
# ----------------- TESTING ----------------- #

def test_primitive_policy():
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    #affordance_model = load_model("graspnet/rgbd_resnet_iter_317452.pth")
    # setup the environment
    env = make_ee_sim_env('primitive')
    # Get mocap positions
    # Get left gripper pose
    ts = env.reset()
    camera_info = {"left": extract_intrinsics(ts.observation["camera_info"]["left"]), "right": extract_intrinsics(ts.observation["camera_info"]["right"])}

    objects=["O02@0094@00001", "O02@0094@00004", "S20005"]
    policy = PrimitivesPolicy(inject_noise, objects, camera_info, None)

    with open("annotations.json", 'r') as f:
        annotations = json.load(f)

    annotations_keys = list(annotations.keys())

    # Extract intrinsics
    finished = False
    curr_primitive = None
    if onscreen_render:
        fig, ax = plt.subplots(3, 2, figsize=(10, 10))  # Two subplots side by side
        plt_img_teleop = ax[0][0].imshow(ts.observation['images']['teleoperator_pov'])
        plt_img_collab = ax[0][1].imshow(ts.observation['images']['collaborator_pov'])
        plt_img_left = ax[1][0].imshow(ts.observation['images']['left_wrist'])
        plt_img_right = ax[1][1].imshow(ts.observation['images']['right_wrist'])
        plt_img_left_depth = ax[2][0].imshow(ts.observation['images']['left_depth'])
        plt_img_right_depth = ax[2][1].imshow(ts.observation['images']['right_depth'])
        
        ax[0][0].set_title("Teleoperator View")
        ax[0][1].set_title("Collaborator View")
        ax[1][0].set_title("Left Wrist View")
        ax[1][1].set_title("Right Wrist View")
        ax[2][0].set_title("Left Wrist Depth")
        ax[2][1].set_title("Right Wrist Depth")

        
        plt.ion()  # Interactive mode to update images dynamically
        plt_img_teleop.set_data(ts.observation['images']['teleoperator_pov'])
        plt_img_collab.set_data(ts.observation['images']['collaborator_pov'])
        plt_img_left.set_data(ts.observation['images']['left_wrist'])
        plt_img_right.set_data(ts.observation['images']['right_wrist'])
        plt_img_left_depth.set_data(ts.observation['images']['left_depth'])
        plt_img_right_depth.set_data(ts.observation['images']['right_depth'])

        fig.savefig("evolution.png")
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
            if p_index >= len(annotations_keys):
                finished = True

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
                plt_img_teleop.set_data(ts.observation['images']['teleoperator_pov'])
                plt_img_collab.set_data(ts.observation['images']['collaborator_pov'])
                plt_img_left.set_data(ts.observation['images']['left_wrist'])
                plt_img_right.set_data(ts.observation['images']['right_wrist'])
                plt_img_left_depth.set_data(ts.observation['images']['left_depth'])
                plt_img_right_depth.set_data(ts.observation['images']['right_depth'])
                fig.savefig("evolution.png")
                plt.pause(0.02)  # Pause to allow the figure to update
                
            # TODO: Check if it needs also p_{r,l} = None
            finished_primitives = (len(primitives_to_execute["left"]) == 0 and len(primitives_to_execute["right"]) == 0) and (terminated_left and terminated_right)
        
        curr_primitive = None
    plt.close()


if __name__ == '__main__':
    test_primitive_policy()

