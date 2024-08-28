import torch.nn as nn
import torch
import os
from collections import deque

from pretrained_encoder import load_pretrained_model


class MLPActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, final='tanh'):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim[0]), nn.ReLU(inplace=True)]
        for ind in range(0, len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[ind], hidden_dim[ind+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim[-1], out_dim))
        if final == 'tanh':
            layers.append(nn.Tanh())
        elif final == 'softmax':
            layers.append(nn.Softmax(dim=1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ImitationLearningPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.joint_action_dim = config.action_dim['joint_ac_dim']
        self.gripper_action_dim = config.action_dim['gripper_ac_dim']
        self.joint_actor = MLPActor(config.input_dim, config.hidden_dim, self.joint_action_dim, 'tanh')
        self.gripper_actor = MLPActor(config.input_dim, config.hidden_dim, self.gripper_action_dim, 'none')
        self.device = config.device
        self.joint_actor.to(config.device)
        self.gripper_actor.to(config.device)
        self.optimizer = torch.optim.Adam(list(self.joint_actor.parameters())+
                                          list(self.gripper_actor.parameters()), lr=1e-4)

        self.encoder, self.embedding_dim, self.transform = \
            load_pretrained_model(encoder_name=config.encoder_name, root_dir=config.root_dir)
        self.encoder.to(self.device)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.frame_stack = config.frame_stack
        self.memory_frames_queue = None

    def act(self, obs, start):
        if start:
            # at the beginning of episode, reinitialize the memory bank
            self.memory_frames_queue = deque(maxlen=self.frame_stack)
        with torch.no_grad():
            feature = self.encoder(self.transform(obs).to(self.device))
            self.memory_frames_queue.append(feature)
            for frame_index in range(1, self.frame_stack):
                # add feature vectors of past frames
                index_in_the_queue = max(len(self.memory_frames_queue) - 1 - frame_index, 0)
                feature = torch.concat((feature, self.memory_frames_queue[index_in_the_queue]), dim=-1)
            joint_action = self.joint_actor(feature)
            gripper_action = self.gripper_actor(feature)
            gripper_action = (gripper_action[:, 0] < gripper_action[:, 1]).float().unsqueeze(0)
            # the last action is used for controlling the gripper, should be mapped to (0, 1)
            action = torch.cat((joint_action, gripper_action), dim=-1)

        return action.squeeze(0).cpu().numpy()

    def forward(self, batch):
        obs, action = batch
        obs = obs.to(self.device)
        action = action.float().to(self.device)
        pred_joint_action = self.joint_actor(obs)
        pred_gripper_action = self.gripper_actor(obs)
        joint_actor_loss = nn.functional.mse_loss(pred_joint_action, action[:, :self.joint_action_dim])
        gripper_actor_loss = nn.functional.cross_entropy(pred_gripper_action, action[:, self.joint_action_dim:])
        loss = joint_actor_loss+gripper_actor_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        self.joint_actor.train()
        self.gripper_actor.train()

    def eval(self):
        self.joint_actor.eval()
        self.gripper_actor.eval()
