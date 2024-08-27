from pretrained_encoder import load_pretrained_model
from torch.utils.data import Dataset
import torch
import os
import PIL.Image as Image
import numpy as np
import pickle
from tqdm import tqdm
from collections import deque


class PrecomputedFeatureDataset(Dataset):
    def __init__(self, config):
        self.root_dir = config.root_dir
        self.folder = config.data_folder
        self.img_type = config.img_type
        self.device = config.device
        self.frame_stack = config.frame_stack
        self.encoder, self.embedding_dim, self.transform = \
            load_pretrained_model(encoder_name=config.encoder_name,
                                  checkpoint_path=os.path.join(config.root_dir, config.checkpoint_path))
        self.encoder.to(self.device)

        self.obs_dim = None
        self.action_dim = None

        self.episodes_obs, self.episodes_actions, self.total_num = self.prepare_data()

    def prepare_data(self):
        episodes_path = os.listdir(os.path.join(self.root_dir, self.folder))
        num_episode = len(episodes_path)
        total_num = [0]*(num_episode+1)

        episodes_obs = []
        episodes_actions = []
        for i in tqdm(range(num_episode)):
            memory_frames_queue = deque(maxlen=self.frame_stack)
            total_num[i+1] = total_num[i]
            images_in_one_episode = []
            images_path = os.path.join(self.root_dir, self.folder, 'episode'+str(i), self.img_type)
            image_file = os.listdir(images_path)
            for img_index in range(len(image_file)):
                im = Image.open(os.path.join(images_path, str(img_index)+'.png'))
                with torch.no_grad():
                    processed_feature = self.encoder(self.transform(np.array(im)).to(self.device))
                    processed_feature = processed_feature.squeeze(0).cpu().numpy()
                memory_frames_queue.append(processed_feature)
                for frame_index in range(1, self.frame_stack):
                    # add feature vectors of past frames
                    index_in_the_queue = max(len(memory_frames_queue)-1-frame_index, 0)
                    processed_feature = np.concatenate((processed_feature, memory_frames_queue[index_in_the_queue]))
                if self.obs_dim is None:
                    self.obs_dim = processed_feature.shape[0]
                images_in_one_episode.append(processed_feature)
                total_num[i+1] += 1
            episodes_obs.append(np.stack(images_in_one_episode, axis=0))

            actions_in_one_episode = []
            with open(os.path.join(self.root_dir, self.folder, 'episode'+str(i), 'low_dim_obs.pkl'), "rb") as f:
                demo = pickle.load(f)
            for obs in demo:
                joint_action = (obs.joint_velocities).astype(float)
                gripper_action = np.array([obs.gripper_open])
                action = np.concatenate([joint_action, gripper_action])
                if self.action_dim is None:
                    self.action_dim = action.shape[0]
                actions_in_one_episode.append(action)
            episodes_actions.append(np.stack(actions_in_one_episode, axis=0))

            assert episodes_obs[-1].shape[0] == episodes_actions[-1].shape[0]

        print('Finish loading the data')
        print('Num of episodes: ' + str(len(total_num)-1))
        print('Obs dim: ' + str(self.obs_dim))
        print('Action dim: ' + str(self.action_dim))
        return episodes_obs, episodes_actions, total_num

    def __len__(self):
        return self.total_num[-1]

    def __getitem__(self, index):
        for i in range(len(self.total_num)-1):
            if self.total_num[i] <= index < self.total_num[i+1]:
                return self.episodes_obs[i][index-self.total_num[i]], self.episodes_actions[i][index-self.total_num[i]]
