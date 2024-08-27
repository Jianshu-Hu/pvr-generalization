import torch.nn as nn
import torch
import os
from collections import deque

from pretrained_encoder import load_pretrained_model


class MLPActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim[0]), nn.ReLU(inplace=True)]
        for ind in range(0, len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[ind], hidden_dim[ind+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim[-1], out_dim))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ImitationLearningPolicy():
    def __init__(self, config):
        self.actor = MLPActor(config.input_dim, config.hidden_dim, config.action_dim)
        self.device = config.device
        self.actor.to(config.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.encoder, self.embedding_dim, self.transform = \
            load_pretrained_model(encoder_name=config.encoder_name,
                                  checkpoint_path=os.path.join(config.root_dir, config.checkpoint_path))
        self.encoder.to(self.device)

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
            action = self.actor(feature)
            # the last action is used for controlling the gripper, should be mapped to (0, 1)
            action[:, -1] = (action[:, -1] > 0).float()

        return action.squeeze(0).cpu().numpy()

    def update(self, batch):
        obs, action = batch
        obs = obs.to(self.device)
        action = action.float().to(self.device)
        pred_action = self.actor(obs)
        # the last action is used for controlling the gripper, should be mapped from (0, 1) to (-1, 1)
        action[:, -1] = torch.where(action[:, -1] == 0, -1.0, 1.0)
        # pred_action_new = torch.clone(pred_action)
        # pred_action_new[:, -1] = (pred_action[:, -1] > 0).float()
        loss = nn.functional.mse_loss(action, pred_action)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
