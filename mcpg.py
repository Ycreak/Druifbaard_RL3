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

class Policy_Network(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size, alpha):
        """Initialize the policy network layers

        Args:
            num_inputs (int): the number of elements of the input
            num_actions (int): the number of elements of the output
            hidden_size (int): the number of elements of the hidden state
            alpha (int): the learning rate
        """ 

        super(Policy_Network, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        """Return the probabilities of a state

        Args:
            state (array of ints): state of the cartpole

        Returns:
            [array of ints]: probabilities of the state
        """  

        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)

        return x 
    
    def get_action(self, state):
        """Returns the best action to take for the state

        Args:
            state (array of ints): state of the cartpole

        Returns:
            highest_prob_action (bool): best action to take
            log_prob: the logarithm of the probability of the highest_prob_action 
        """

        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])

        return highest_prob_action, log_prob

class Mcpg:
    """Class for the Monte-Carlo Policy Gradients
    """    
    def main(self, gym, exp, cart, alpha, gamma, iterations, max_steps, hidden_size):
        """Main function of the MCPG class

        Args:
            gym: Gym toolkit
            exp (Experiment_episode_timesteps object): to take care of recording results
            cart (Cart object): the cart
            alpha (int): the learning rate
            gamma (int): the dicount factor
            iterations (int): the amount of episodes to run 
            max_steps (int): the maximum steps for an episode
            hidden_size (int): the number of elements of the hidden state

        Returns:
            [type]: [description]
        """        

        env = gym.make('CartPole-v0')
        policy_net = Policy_Network(env.observation_space.shape[0], env.action_space.n, hidden_size, alpha)
        
        print_per_n = 100
        numsteps = []
        avg_numsteps = []
        all_rewards = []

        total_timesteps = 0
        total_timesteps_n_episodes = 0

        for episode in range(int(iterations)):
            state = env.reset()
            log_probs = []
            rewards = []
            steps = 0
            go_on = True

            while go_on:

                # env.render()
                action, log_prob = policy_net.get_action(state)

                new_state, reward, done, _ = env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)

                state = new_state

                steps += 1
                total_timesteps += 1
                total_timesteps_n_episodes += 1

                if done or steps == max_steps:
                    self.update_policy(policy_net, steps, log_probs, gamma)
                    go_on = False
                
            # Print statistics
            if episode % print_per_n == 0:
                avg_timesteps = total_timesteps / (episode + 1) 

                print("Episode {} finished after {} timesteps".format(episode, steps + 1))
                print("Average timesteps {}".format(avg_timesteps))
                print("Average timesteps of last {} episodes: {}\n".format(print_per_n, total_timesteps_n_episodes / print_per_n))
                exp.Episode_time(episode, avg_timesteps, (total_timesteps_n_episodes / print_per_n))
            
                total_timesteps_n_episodes = 0
        
        return exp.df
    
    def update_policy(self, policy_network, steps, log_probs, gamma):
        """Update the policy network with the logged results

        Args:
            policy_network (Policy_Network object): the policy network
            steps (int): the number of steps taken in the episode
            log_probs (array): the logarithms of the probabilities of the highest_prob_actions
            gamma (int): the dicount factor
        """        
        discounted_rewards = []

        #Since all rewards are 1 and the number of rewards equals the number of steps + 1
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
    
