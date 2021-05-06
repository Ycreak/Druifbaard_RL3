import torch  
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys

def match_action(index):
    assert index >= 0 and index < 8
    if index == 0:
        return [0,0,0]
    elif index == 1:
        return [0,0,1]
    elif index == 2:
        return [0,1,0]
    elif index == 3:
        return [0,1,1]
    elif index == 4:
        return [1,0,0]
    elif index == 5:
        return [1,0,1]
    elif index == 6:
        return [1,1,0]
    elif index == 7:
        return [1,1,1]

class Policy_Network(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, alpha):
        super(Policy_Network, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x 
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

class Mcpg:
    def main(self, gym, gyms, exp, cart, alpha, gamma, iterations, max_steps, hidden_size):
        env = gym.make('SlimeVolley-v0')
        policy_net = Policy_Network(12, 8, hidden_size, alpha)
        
        print_per_n = 100
        numsteps = []
        avg_numsteps = []
        all_rewards = []

        total_timesteps = 0
        total_timesteps_n_episodes = 0

        total_reward = 0
        total_reward_n_episodes = 0

        for episode in range(int(iterations)):
            state = env.reset()
            log_probs = []
            rewards = []
            steps = 0
            go_on = True

            while go_on:

                # env.render()
                action, log_prob = policy_net.get_action(state)

                new_state, reward, done, _ = env.step(match_action(index=action))

                log_probs.append(log_prob)
                rewards.append(reward)

                state = new_state

                steps += 1
                total_timesteps += 1
                total_timesteps_n_episodes += 1

                total_reward += reward
                total_reward_n_episodes += reward

                if done or steps == max_steps:
                    self.update_policy(policy_net, steps, log_probs, gamma)
                    go_on = False
                
            # Print statistics
            if episode % print_per_n == 0:
                avg_timesteps = total_timesteps / (episode + 1) 
                avg_reward = total_reward / (episode + 1)

                print("Episode {} finished after {} timesteps".format(episode, steps + 1))
                print("Average timesteps {} -- reward {}".format(avg_timesteps, avg_reward))
                print("Average timesteps of last {} episodes: {} -- reward {}\n".format(print_per_n, total_timesteps_n_episodes / print_per_n, total_reward_n_episodes / print_per_n))
                exp.Episode_time(episode, avg_timesteps, (total_timesteps_n_episodes / print_per_n))
            
                total_timesteps_n_episodes = 0
                total_reward_n_episodes = 0
        
        return exp.df
    
    def update_policy(self, policy_network, steps, log_probs, gamma):
        
        discounted_rewards = []

        for step in range(steps + 1):
            Gt = 0 
            pw = 0
            for r in range(step, steps + 1):
                Gt = Gt + gamma ** pw
                pw = pw + 1

            discounted_rewards.append(Gt)
            
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
        
        policy_network.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        policy_network.optimizer.step()
    
