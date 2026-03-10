import os  
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

class Diffusion_QL(object):
    def __init__(self, state_dim, action_dim, max_action, device, discount, tau, 
                 max_q_backup=False, eta=1.0, beta_schedule='linear', n_timesteps=100,
                 ema_decay=0.995, step_start_ema=1000, update_ema_every=5, lr=3e-4, 
                 lr_decay=False, lr_maxt=1000, grad_norm=1.0):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, 
                               max_action=max_action, beta_schedule=beta_schedule, n_timesteps=n_timesteps).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta
        self.device = device
        self.max_q_backup = max_q_backup
        self.grad_norm = grad_norm

    def step_ema(self):
        if self.step < self.step_start_ema: return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        
        for _ in range(iterations):
            if isinstance(replay_buffer, dict):
                state = replay_buffer['observations']
                action = replay_buffer['actions']
                next_state = replay_buffer['next_observations']
                # Correction cruciale des dimensions pour éviter le Warning
                reward = replay_buffer['rewards'].reshape(-1, 1)
                not_done = (1 - replay_buffer['terminals']).reshape(-1, 1)
            else:
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # Q Training
            current_q1, current_q2 = self.critic(state, action)
            next_action = self.ema_model(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = (reward + not_done * self.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Policy Training
            bc_loss = self.actor.loss(action, state)
            new_action = self.actor(state)
            q1_new, q2_new = self.critic(state, new_action)
            q_loss = - torch.min(q1_new, q2_new).mean()
            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.step % self.update_ema_every == 0: self.step_ema()
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1
            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        os.makedirs(dir, exist_ok=True)
        suffix = f"_{id}" if id else ""
        torch.save(self.actor.state_dict(), f'{dir}/actor{suffix}.pth')
        torch.save(self.critic.state_dict(), f'{dir}/critic{suffix}.pth')