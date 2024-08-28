import argparse
from omegaconf import OmegaConf

from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import random
from datetime import datetime
from PIL import Image
import csv

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


def eval(task_env, config, policy, log_dir):
    # TODO: find out which part is time-consuming and accelerate it
    sum_reward_list = []
    for ep_idx in range(config.eval_episodes):
        _, obs = task_env.reset()
        sum_reward = 0
        for time_index in range(config.max_length):
            obs = retrieve_obs(obs, config.data_cfg)
            if config.save_results:
                new_log_dir = os.path.join(log_dir, 'images_episode_'+str(ep_idx))
                if not os.path.exists(new_log_dir):
                    os.mkdir(new_log_dir)
                img = Image.fromarray(obs)
                img.save(new_log_dir+'/'+str(time_index)+'.png')
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
    config.policy.action_dim = {'joint_ac_dim': dataset.joint_action_dim, 'gripper_ac_dim': dataset.gripper_action_dim}
    policy = ImitationLearningPolicy(config.policy)
    trainable_param_size = 0
    frozen_param_size = 0
    num_trainable = 0
    num_frozen = 0
    for param in policy.parameters():
        if param.requires_grad:
            trainable_param_size += param.nelement() * param.element_size()
            num_trainable += param.numel()
        else:
            frozen_param_size += param.nelement() * param.element_size()
            num_frozen += param.numel()
    # buffer_size = 0
    # for buffer in policy.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()
    print('Actor size: {:.3f}MB'.format(trainable_param_size / (1024**2)))
    print('Pretrained encoder size: {:.3f}MB'.format(frozen_param_size / (1024 ** 2)))
    print('Number of trainable parameters of the policy: ' + str(num_trainable))
    policy.train()

    print("========================================")
    print("Train")
    print("========================================")
    # save the training and evaluation results to csv for quick check
    csv_result = [
        ['Epoch', 'Loss', 'Score', 'Highest_score']
    ]
    with open(config.specific_log_dir+'/evaluation_scores.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_result)
    score = 0
    highest_score = 0
    for epoch in tqdm(range(config["epoch"])):
        avg_loss = 0
        for ind, batch in enumerate(dataloader):
            loss = policy(batch)
            avg_loss = (avg_loss*ind+loss)/(ind+1)
        print('avg loss in epoch ' + str(epoch) + ': ' + str(avg_loss))

        if (epoch+1) % config.eval.eval_freq == 0:
            print("========================================")
            print("Eval at " + str(epoch+1) + " epoch")
            print("========================================")
            policy.eval()
            score = eval(task_env, config.eval, policy, config.specific_log_dir)
            policy.train()
            if score > highest_score:
                highest_score = score

        csv_result.append([epoch, avg_loss, score, highest_score])
        with open(config.specific_log_dir+'/evaluation_scores.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_result)

    rlbench_env.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", type=str, default="cfgs/config.yaml", help="config file")
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--encoder", type=str, default="vc1_base",
                        choices={'dinov2_base', "vc1_base", "vc1_large"})
    parser.add_argument("--env", type=str, default="open_drawer")
    args = parser.parse_args()

    # set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("========================================")
    print("Job Configuration")
    print("========================================")
    config = OmegaConf.load(args.config)

    config.encoder_name = args.encoder
    config.task_name = args.env

    # create log dir
    current_time = datetime.now()
    log_dir = os.path.join(config.root_dir, config.log_dir, args.name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    specific_log_dir = os.path.join(config.root_dir, config.log_dir, args.name,
                                    current_time.strftime("%Y-%m-%d-%H-%M-%S")+'-seed-'+str(args.seed))
    os.mkdir(specific_log_dir)
    print("files log dir:" + specific_log_dir)
    config.specific_log_dir = specific_log_dir

    print(config)

    train(config)
