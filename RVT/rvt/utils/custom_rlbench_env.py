# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# from rvt.libs.peract.helpers.custom_rlbench_env import CustomMultiTaskRLBenchEnv
import os
from typing import Type, List, Dict, Any
import json
from omegaconf import DictConfig, OmegaConf

import numpy as np
from rlbench import ObservationConfig, Environment
from rlbench.backend.exceptions import InvalidActionError
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task

from yarr.envs.env import Env, MultiTaskEnv
from yarr.agents.agent import ActResult, VideoSummary, TextSummary
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition
from yarr.utils.process_str import change_case
from yarr.envs.rlbench_env import (
    ROBOT_STATE_KEYS,
    _extract_obs,
    _get_cam_observation_elements,
    _observation_elements
)

from clip import tokenize
from pyrep.const import RenderMode
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.objects import VisionSensor, Dummy

from colosseum import (
    ASSETS_CONFIGS_FOLDER,
    ASSETS_JSON_FOLDER,
    TASKS_PY_FOLDER,
    TASKS_TTM_FOLDER,
)
from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.tools.dataset_generator import get_spreadsheet_config, get_variation_name


class MultiTaskRLBenchEnv(MultiTaskEnv):

    def __init__(self,
                 task_classes: List[Type[Task]],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 dataset_root: str = '',
                 channels_last=False,
                 headless=True,
                 swap_task_every: int = 1,
                 include_lang_goal_in_obs=False,
                 base_cfg_name=None,
                 task_class_variation_idx=None):
        super(MultiTaskRLBenchEnv, self).__init__()

        self._task_classes = task_classes
        self._observation_config = observation_config
        self._channels_last = channels_last
        self._include_lang_goal_in_obs = include_lang_goal_in_obs
        # self._rlbench_env = Environment(
        #     action_mode=action_mode, obs_config=observation_config,
        #     dataset_root=dataset_root, headless=headless)
        self._task = None
        self._task_name = ''
        self._lang_goal = 'unknown goal'
        self._swap_task_every = swap_task_every

        self._episodes_this_task = 0
        self._active_task_id = -1

        self._task_name_to_idx = {change_case(tc.__name__):i for i, tc in enumerate(self._task_classes)}
        self._base_cfg_name = base_cfg_name
        self._task_class_variation_idx = task_class_variation_idx
        self._action_mode = action_mode
        self._observation_config = observation_config
        self._dataset_root = dataset_root
        self._headless = headless

    def _set_new_task(self, shuffle=False):
        if shuffle:
            self._active_task_id = np.random.randint(0, len(self._task_classes))
        else:
            self._active_task_id = (self._active_task_id + 1) % len(self._task_classes)
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)

    def set_task(self, task_name: str):
        self._active_task_id = self._task_name_to_idx[task_name]
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)

        descriptions, _ = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant

    def extract_obs(self, obs: Observation):
        extracted_obs = _extract_obs(obs, self._channels_last, self._observation_config)
        if self._include_lang_goal_in_obs:
            extracted_obs['lang_goal_tokens'] = tokenize([self._lang_goal])[0].numpy()
        return extracted_obs

    def launch(self):
        base_cfg_path = os.path.join(ASSETS_CONFIGS_FOLDER, f"{self._base_cfg_name[self._active_task_id]}.yaml")
        if os.path.exists(base_cfg_path):
            with open(base_cfg_path, 'r') as f:
                base_cfg = OmegaConf.load(f)

        collection_cfg_path: str = (
        os.path.join(ASSETS_JSON_FOLDER, base_cfg.env.task_name) + ".json"
        )
        collection_cfg: Optional[Any] = None
        with open(collection_cfg_path, "r") as fh:
            collection_cfg = json.load(fh)

        if collection_cfg is None:
            return 1

        if "strategy" not in collection_cfg:
            return 1

        num_spreadsheet_idx = len(collection_cfg["strategy"])

        if self._task_class_variation_idx != None:
            full_config = get_spreadsheet_config(
                        base_cfg,
                        collection_cfg,
                        self._task_class_variation_idx[self._active_task_id],
                    )
            _, env_cfg = full_config.data, full_config.env
        else:
            env_cfg = None

        self._rlbench_env = EnvironmentExt(
            action_mode=self._action_mode, obs_config=self._observation_config,
            path_task_ttms=TASKS_TTM_FOLDER,
            dataset_root=self._dataset_root, headless=self._headless, env_config=env_cfg,)
        self._rlbench_env

        self._rlbench_env.launch()
        self._set_new_task()

    def shutdown(self):
        self._rlbench_env.shutdown()

    def reset(self) -> dict:
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        descriptions, obs = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant
        extracted_obs = self.extract_obs(obs)

        return extracted_obs

    def step(self, action: np.ndarray) -> Transition:
        obs, reward, terminal = self._task.step(action)
        obs = self.extract_obs(obs)
        return Transition(obs, reward, terminal)

    @property
    def observation_elements(self) -> List[ObservationElement]:
        return _observation_elements(self._observation_config, self._channels_last)

    @property
    def action_shape(self):
        return (self._rlbench_env.action_size, )

    @property
    def env(self) -> Environment:
        return self._rlbench_env

    @property
    def num_tasks(self) -> int:
        return len(self._task_classes)


class CustomMultiTaskRLBenchEnv(MultiTaskRLBenchEnv):

    def __init__(self,
                 task_classes: List[Type[Task]],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 episode_length: int,
                 dataset_root: str = '',
                 channels_last: bool = False,
                 reward_scale=100.0,
                 headless: bool = True,
                 swap_task_every: int = 1,
                 time_in_state: bool = False,
                 include_lang_goal_in_obs: bool = False,
                 record_every_n: int = 20,
                 base_cfg_name=None,
                 task_class_variation_idx=None):

        super(CustomMultiTaskRLBenchEnv, self).__init__(
            task_classes, observation_config, action_mode, dataset_root,
            channels_last, headless=headless, swap_task_every=swap_task_every,
            include_lang_goal_in_obs=include_lang_goal_in_obs,
            base_cfg_name=base_cfg_name,
            task_class_variation_idx=task_class_variation_idx)
        self._reward_scale = reward_scale
        self._episode_index = 0
        self._record_current_episode = False
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._episode_length = episode_length
        self._time_in_state = time_in_state
        self._record_every_n = record_every_n
        self._i = 0
        self._error_type_counts = {
            'IKError': 0,
            'ConfigurationPathError': 0,
            'InvalidActionError': 0,
        }
        self._last_exception = None

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomMultiTaskRLBenchEnv, self).observation_elements
        for oe in obs_elems:
            if oe.name == 'low_dim_state':
                oe.shape = (oe.shape[0] - 7 * 3 + int(self._time_in_state),)  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
        return obs_elems

    def extract_obs(self, obs: Observation, t=None, prev_action=None):
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        joint_pos = obs.joint_positions
        obs.gripper_pose = None
        # obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.wrist_camera_matrix = None
        obs.joint_positions = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0., 0.04)

        obs_dict = super(CustomMultiTaskRLBenchEnv, self).extract_obs(obs)

        if self._time_in_state:
            time = (1. - ((self._i if t is None else t) / float(
                self._episode_length - 1))) * 2. - 1.
            obs_dict['low_dim_state'] = np.concatenate(
                [obs_dict['low_dim_state'], [time]]).astype(np.float32)

        obs.gripper_matrix = grip_mat
        # obs.gripper_pose = grip_pose
        obs.joint_positions = joint_pos
        obs.gripper_pose = grip_pose
        # obs_dict['gripper_pose'] = grip_pose
        return obs_dict

    def launch(self):
        super(CustomMultiTaskRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_base = Dummy('cam_cinematic_base')
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def reset(self) -> dict:
        self._i = 0
        self._previous_obs_dict = super(CustomMultiTaskRLBenchEnv, self).reset()
        self._record_current_episode = (
                self.eval and self._episode_index % self._record_every_n == 0)
        self._episode_index += 1
        self._recorded_images.clear()
        return self._previous_obs_dict

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _my_callback(self):
        if self._record_current_episode:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self._recorded_images.append(cap)

    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10, ) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

    def step(self, act_result: ActResult) -> Transition:
        action = act_result.action
        success = False
        obs = self._previous_obs_dict  # in case action fails.

        try:
            obs, reward, terminal = self._task.step(action)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0

            if isinstance(e, IKError):
                self._error_type_counts['IKError'] += 1
            elif isinstance(e, ConfigurationPathError):
                self._error_type_counts['ConfigurationPathError'] += 1
            elif isinstance(e, InvalidActionError):
                self._error_type_counts['InvalidActionError'] += 1

            self._last_exception = e

        summaries = []
        self._i += 1
        if ((terminal or self._i == self._episode_length) and
                self._record_current_episode):
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            task_name = change_case(self._task._task.__class__.__name__)
            summaries.append(VideoSummary(
                'episode_rollout_' + ('success' if success else 'fail') + f'/{task_name}',
                vid, fps=30))

            # error summary
            error_str = f"Errors - IK : {self._error_type_counts['IKError']}, " \
                        f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, " \
                        f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
            if not success and self._last_exception is not None:
                error_str += f"\n Last Exception: {self._last_exception}"
                self._last_exception = None

            summaries.append(TextSummary('errors', f"Success: {success} | " + error_str))
        return Transition(obs, reward, terminal, summaries=summaries)

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0
        # super(CustomMultiTaskRLBenchEnv, self).reset()

        # if variation_number == -1:
        #     self._task.sample_variation()
        # else:
        #     self._task.set_variation(variation_number)

        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i)[0]

        self._task.set_variation(d.variation_number)
        _, obs = self._task.reset_to_demo(d)
        self._lang_goal = self._task.get_task_descriptions()[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
                self.eval and self._episode_index % self._record_every_n == 0)
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict


class CustomMultiTaskRLBenchEnv2(CustomMultiTaskRLBenchEnv):
    def __init__(self, *args, **kwargs):
        super(CustomMultiTaskRLBenchEnv2, self).__init__(*args, **kwargs)

    def reset(self) -> dict:
        super().reset()
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        return self._previous_obs_dict

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0

        # self._task.set_variation(-1)
        if self._task_class_variation_idx != None:
            self._task.set_variation(0)
            self._task._task.task_path = self._task._task.name + f"_{str(self._task_class_variation_idx[self._active_task_id])}"
        else:
            self._task.set_variation(-1)

        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i
        )[0]

        self._task.set_variation(d.variation_number)
        # TODO: there is a problem here. Although the random seed is set the same as the random seed in the demo, the
        #  generated observation is still different from the demo.
        desc, obs = self._task.reset_to_demo(d)
        self._lang_goal = desc[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict
