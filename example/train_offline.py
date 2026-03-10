import sys
import os
import h5py
import torch
import numpy as np

# Ajouter le dossier actuel au path pour les imports
sys.path.append(os.getcwd())
from agents.ql_diffusion import Diffusion_QL

# 1. Chemins
dataset_path = '../dataset/easycarla_offline_dataset.hdf5'
save_dir = 'params_dql_new'

# 2. Chargement du Dataset
print("Chargement du dataset en cours...")
with h5py.File(dataset_path, 'r') as f:
    observations = f['observations'][:]
    actions = f['actions'][:]
    rewards = f['rewards'][:]
    next_observations = f['next_observations'][:]
    dones = f['done'][:]

# 3. Initialisation Agent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = Diffusion_QL(state_dim=307, action_dim=3, max_action=1.0, 
                     device=device, discount=0.99, tau=0.005)

# 4. Boucle d'entraînement
epochs = 50
steps_per_epoch = 1000
batch_size = 256

print(f"Début de l'entraînement sur {device}...")

for epoch in range(1, epochs + 1):
    epoch_metrics = []
    
    for step in range(steps_per_epoch):
        idx = np.random.randint(0, len(observations), size=batch_size)
        
        batch = {
            'observations': torch.tensor(observations[idx], dtype=torch.float32).to(device),
            'actions': torch.tensor(actions[idx], dtype=torch.float32).to(device),
            'rewards': torch.tensor(rewards[idx], dtype=torch.float32).to(device),
            'next_observations': torch.tensor(next_observations[idx], dtype=torch.float32).to(device),
            'terminals': torch.tensor(dones[idx], dtype=torch.float32).to(device)
        }

        # Entraînement et récupération des dictionnaires de metrics
        m = agent.train(batch, iterations=1)
        epoch_metrics.append(m)

    # Calcul des moyennes (m est un dictionnaire de listes)
    avg_bc = np.mean([np.mean(m['bc_loss']) for m in epoch_metrics])
    avg_ql = np.mean([np.mean(m['ql_loss']) for m in epoch_metrics])
    avg_critic = np.mean([np.mean(m['critic_loss']) for m in epoch_metrics])

    agent.save_model(save_dir, id=f"epoch_{epoch}")
    
    print(f"Époque {epoch}/{epochs} | BC Loss: {avg_bc:.4f} | QL Loss: {avg_ql:.4f} | Critic: {avg_critic:.4f}")

print("Fini !")