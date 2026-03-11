import os
import h5py
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from agents.ql_diffusion import Diffusion_QL

# --- 1. CONFIGURATION DES CHEMINS ---
if os.path.exists('/kaggle/input'):
    # Chemin spécifique à ton dataset sur Kaggle
    dataset_path = '/kaggle/input/datasets/sarasouhail/dataset/easycarla_offline_dataset.hdf5'
    save_dir = '/kaggle/working/params_dql_new'
else:
    # Chemin pour ton usage local
    dataset_path = '../dataset/easycarla_offline_dataset.hdf5'
    save_dir = 'params_dql_new'

os.makedirs(save_dir, exist_ok=True)

# --- 2. CHARGEMENT DU DATASET ---
print(f"Chargement du dataset : {dataset_path}")
with h5py.File(dataset_path, 'r') as f:
    observations = f['observations'][:]
    actions = f['actions'][:]
    next_observations = f['next_observations'][:]
    rewards = f['rewards'][:]
    dones = f['done'][:]

print(f"Dataset chargé avec succès ({len(observations)} échantillons).")

# --- 3. INITIALISATION DE L'AGENT ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --- INITIALISATION DE L'AGENT (MODIFIÉE) ---
agent = Diffusion_QL(
    state_dim=307, 
    action_dim=3, 
    max_action=1.0, 
    device=device, 
    discount=0.95,   # Passé de 0.99 à 0.95 pour stabiliser le Critique
    tau=0.001        # Passé de 0.005 à 0.001 pour des mises à jour plus douces
)

# --- 4. CONFIGURATION DE L'ENTRAÎNEMENT ---
epochs = 20
steps_per_epoch = 1000
batch_size = 256
history = [] # Pour stocker les données des courbes

print(f"Début de l'entraînement sur {device}...")

# --- 5. BOUCLE D'ENTRAÎNEMENT ---
for epoch in range(1, epochs + 1):
    epoch_metrics = []
    
    # Barre de progression pour chaque époque
    pbar = tqdm(range(steps_per_epoch), desc=f"Époque {epoch}/{epochs}")
    
    for step in pbar:
        # Échantillonnage aléatoire
        idx = np.random.randint(0, len(observations), size=batch_size)
        
        batch = {
            'observations': torch.tensor(observations[idx], dtype=torch.float32).to(device),
            'actions': torch.tensor(actions[idx], dtype=torch.float32).to(device),
            'rewards': torch.tensor(rewards[idx], dtype=torch.float32).to(device),
            'next_observations': torch.tensor(next_observations[idx], dtype=torch.float32).to(device),
            'terminals': torch.tensor(dones[idx], dtype=torch.float32).to(device)
        }

        # Entraînement
        m = agent.train(batch, iterations=1)
        epoch_metrics.append(m)
        
        # Mise à jour de la barre de progression (optionnel)
        if step % 100 == 0:
            pbar.set_postfix({'BC': f"{np.mean(m['bc_loss']):.4f}"})

    # Calcul des moyennes de l'époque
    avg_bc = np.mean([np.mean(m['bc_loss']) for m in epoch_metrics])
    avg_ql = np.mean([np.mean(m['ql_loss']) for m in epoch_metrics])
    avg_critic = np.mean([np.mean(m['critic_loss']) for m in epoch_metrics])

    # Enregistrement dans l'historique
    history.append({
        'epoch': epoch,
        'bc_loss': avg_bc,
        'ql_loss': avg_ql,
        'critic_loss': avg_critic
    })
    
    # Sauvegarde immédiate du log CSV (pour voir les courbes même si l'entraînement est coupé)
    pd.DataFrame(history).to_csv(os.path.join(save_dir, 'training_log.csv'), index=False)

    # Sauvegarde du modèle
    agent.save_model(save_dir, id=f"epoch_{epoch}")
    
    print(f"\rÉpoque {epoch} Terminée | BC Loss: {avg_bc:.4f} | QL Loss: {avg_ql:.4f} | Critic: {avg_critic:.4f}")

print(f"Entraînement terminé. Les logs sont dans {save_dir}/training_log.csv")