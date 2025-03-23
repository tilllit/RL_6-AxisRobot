import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from ENVIRONMENT import Env

# Hyperparameter
gamma = 0.99
exp = 0.01
lr = 0.00001

torch.manual_seed(45)


class Pi(nn.Module):
    def __init__(self, in_dim, num_joints=6, num_options=3):
        """
        :param in_dim: Dimension des Zustands (hier: 6 Gelenkwinkel)
        :param num_joints: Anzahl der Gelenke (6)
        :param num_options: Anzahl der Optionen pro Gelenk (3: -0.1, 0.0, +0.1)
        """
        super(Pi, self).__init__()
        self.num_joints = num_joints
        self.num_options = num_options
        out_dim = num_joints * num_options  # 6 * 3 = 18
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
        self.agg_entropies = []

    def forward(self, x):
        pdparam = self.model(x)
        # Umformen in (6, 3) für die 6 Köpffe
        pdparam = pdparam.view(self.num_joints, self.num_options)
        return pdparam

    def act(self, state):
        x = torch.from_numpy(np.array(state, dtype=np.float32))
        logits = self.forward(x)  # Form: (6, 3)
        actions = []
        log_prob_list = []
        entropy_list = []
        for i in range(self.num_joints):
            dist = Categorical(logits=logits[i])
            action_i = dist.sample()
            actions.append(action_i.item())
            log_prob_list.append(dist.log_prob(action_i))
            entropy_list.append(dist.entropy())
        total_log_prob = torch.stack(log_prob_list).sum()
        total_entropy = torch.stack(entropy_list).sum()
        self.log_probs.append(total_log_prob)
        self.agg_entropies.append(total_entropy)
        
        mapping = {0: -0.1, 1: 0.0, 2: 0.1}
        deltas = np.array([mapping[a] for a in actions])
        #print(deltas)
        return deltas

def train(pi, optimizer):
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    
    # Entropien über alle Steps
    total_entropy = torch.stack(pi.agg_entropies).sum() if len(pi.agg_entropies) > 0 else 0

    loss = - (log_probs * rets).sum() - exp * total_entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def main():
    dh_params = [
        [0, 0.15185, 0, 90],
        [0, 0, -0.24355, 0],
        [0, 0, -0.2132, 0],
        [0, 0.13105, 0, 90],
        [0, 0.08535, 0, -90],
        [0, 0.0921, 0, 0]
    ]
    joint_angles = [180, -90, 90, 90, 90, 0]
    env = Env(dh_params, joint_angles)
    
    state = env.reset()
    # Der Zustand besteht hier aus 6 Gelenkwinkeln
    pi = Pi(in_dim=6)  # in_dim entspricht der Anzahl der Gelenkwinkel
    optimizer = optim.Adam(pi.parameters(), lr=lr)

    # Load model
    path = 'model/Neu2REINFORCE30000.pt'
    pi.load_state_dict(torch.load(path))
    
    total_rewards = []
    total_loss = []
    total_mse = []


    epochs = 1000
    for epi in range(epochs):
        state = env.reset()
        env.plotTraj = []
        steps = 700
        for t in range(steps):  # Maximale Schritte pro Episode
            action_deltas = pi.act(state)
            
            # Periodisches Plotten
            plot = False
            if epi < 4000:    
                plot = (epi % 100 == 0 )                        # Alle 100 Episoden
            else:
                plot = (epi+1 == epochs)  or (epi % 1000 == 0 ) # Alle 1000 Episoden & letzte Episode
            
            next_state, reward, terminated, truncated, MSEHistory = env.step(action_deltas, plot=plot)
            done = terminated or truncated
            
            pi.rewards.append(reward)
            state = next_state
            if done:
                break
        

        epoch_mse = np.mean(env.mseHistory) if len(env.mseHistory) > 0 else 0
        total_mse.append(epoch_mse)

        loss = train(pi, optimizer)
        total_loss.append(loss.item())
        total_reward = sum(pi.rewards)
        total_rewards.append(total_reward)


        solved = total_reward > 195.0
        pi.onpolicy_reset()  # Speicher zurücksetzen
        print(f'Episode {epi}, loss: {loss.item()}, total_reward: {total_reward}, epoch MSE: {epoch_mse}, solved: {solved}')

        # Speicht model alle 5000 episoden
        #if (epi % 5000 == 0):
            #torch.save(pi.state_dict(), './model/Neu3REINFORCE%03d.pt'%epi)

    #torch.save(pi.state_dict(), './model/Neu3REINFORCE%03d.pt'%epochs)


    # ------------------ Darstellung der Ergebnisse ------------------
    fig = plt.figure(figsize=(16, 12))

    # Oberes Subplot: Total Rewards und Loss
    plt.subplot(2, 1, 1)
    plt.plot(total_rewards, marker='o', linestyle='-', label='Total Rewards')
    total_loss_divided = [value / 1000 for value in total_loss]
    plt.plot(total_loss_divided, marker='o', linestyle='-', color='red', label='Total Loss')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.title("Total Reward per Episode")
    plt.grid(True)

    # Unteres Subplot: Durchschnittlicher MSE pro Episode
    plt.subplot(2, 1, 2)
    plt.plot(total_mse, marker='o', linestyle='-', label='Epoch MSE')
    plt.xlabel("Episode")
    plt.ylabel("MSE")
    plt.title("Durchschnittlicher MSE pro Episode")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("TestTotalReward_and_EpochMSE.png")
    plt.close(fig)

if __name__ == '__main__':
    main()