import argparse
from omegaconf import OmegaConf

from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import random

from dataset import PrecomputedFeatureDataset
from imitation_learning import ImitationLearningPolicy
from obs_process import retrieve_obs

from colosseum import ASSETS_CONFIGS_FOLDER, TASKS_PY_FOLDER, TASKS_TTM_FOLDER
from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import (
    ObservationConfigExt,
    check_and_make,
    name_to_class,
    save_demo,
)
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import const


def make_env(config):
    task_class = name_to_class(config.task_name, TASKS_PY_FOLDER)
    assert (
            task_class is not None
    ), f"Can't get task-class for task {config.task_name}"

    # load config
    env_config = OmegaConf.load(os.path.join(config.root_dir, config.config_file))
    data_cfg, env_cfg = env_config.data, env_config.env

    rlbench_env = EnvironmentExt(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfigExt(data_cfg),
        headless=True,
        path_task_ttms=TASKS_TTM_FOLDER,
        env_config=env_cfg,
    )

    rlbench_env.launch()

    task_env = rlbench_env.get_task(task_class)
    print(f"Task: {task_env.get_name()}")

    descriptions, _ = task_env.reset()
    print(descriptions)
    print('action shape: ' + str(rlbench_env.action_shape))
    return rlbench_env, task_env


def eval(task_env, config, policy):
    sum_reward_list = []
    for ep_idx in range(config.eval_episodes):
        _, obs = task_env.reset()
        sum_reward = 0
        for time_index in range(config.max_length):
            obs = retrieve_obs(obs, config.data_cfg)
            action = policy.act(obs, time_index == 0)
            obs, reward, terminate = task_env.step(action)
            sum_reward += reward
            if terminate:
                break
        print('episode '+str(ep_idx)+': '+str(sum_reward))
        sum_reward_list.append(sum_reward)
    avg_reward = sum(sum_reward_list)/len(sum_reward_list)
    print(f"// Average Score: {avg_reward}")
    return avg_reward


def train(config):
    print("========================================")
    print("Make env")
    print("========================================")
    # build a eval env
    rlbench_env, task_env = make_env(config.env)
    print("========================================")
    print("Build dataset")
    print("========================================")
    dataset = PrecomputedFeatureDataset(config.dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    print("========================================")
    print("Policy")
    print("========================================")
    config.policy.input_dim = dataset.obs_dim
    config.policy.action_dim = dataset.action_dim
    policy = ImitationLearningPolicy(config.policy)
    param_size = 0
    for param in policy.actor.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in policy.actor.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    print('Model size: {:.3f}MB'.format((param_size + buffer_size) / (1024**2)))
    print('Number of parameters of the policy: ' + str(sum(p.numel() for p in policy.actor.parameters())))
    policy.actor.train()

    # print("========================================")
    # print("Eval initial policy")
    # print("========================================")
    # policy.actor.eval()
    # eval(task_env, config.eval, policy)

    print("========================================")
    print("Train")
    print("========================================")
    score_log = 0
    for epoch in tqdm(range(config["epoch"])):
        avg_loss = 0
        for ind, batch in enumerate(dataloader):
            loss = policy.update(batch)
            avg_loss = (avg_loss*ind+loss)/(ind+1)
        print('avg loss in epoch ' + str(epoch) + ': ' + str(avg_loss))

        if (epoch+1) % config.eval.eval_freq == 0:
            print("========================================")
            print("Eval at " + str(epoch+1) + " epoch")
            print("========================================")
            policy.actor.eval()
            eval(task_env, config.eval, policy)
            policy.actor.train()

    rlbench_env.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", type=str, default="cfgs/config.yaml", help="config file")
    # parser.add_argument("--name", type=str, default="test")
    # parser.add_argument("--encoder", type=str, default="dinov2_base_patch14")
    # parser.add_argument("--env", type=str, default="dmc_walker_stand-v1")
    args = parser.parse_args()

    # set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config = OmegaConf.load(args.config)
    print("========================================")
    print("Job Configuration")
    print("========================================")
    print(config)

    train(config)