from __future__ import absolute_import, division, print_function

import numpy as np
import random 
import pandas
import torch
import copy
from collections import deque

class DQN_Agent:
    
    def __init__(self, seed, layer_sizes, lr, sync_freq, exp_replay_size):
        torch.manual_seed(seed)
        self.q_net = self.build_nn(layer_sizes)
        self.target_net = copy.deepcopy(self.q_net)
        self.q_net
        self.target_net
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        
        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(0.95).float()
        self.experience_replay = deque(maxlen = exp_replay_size)  
        return
        
    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes)-1):
            linear = torch.nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act =    torch.nn.Tanh() if index < len(layer_sizes)-2 else torch.nn.Identity()
            layers += (linear,act)
        return torch.nn.Sequential(*layers)
    
    def get_action(self, state, action_space_len, epsilon):
        # We do not require gradient at this point, because this function will be used either
        # during experience collection or during inference
        with torch.no_grad():
            Qp = self.q_net(torch.from_numpy(state).float())
        Q,A = torch.max(Qp, axis=0)
        A = A if torch.rand(1,).item() > epsilon else torch.randint(0,action_space_len,(1,))
        return A
    
    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q,_ = torch.max(qp, axis=1)    
        return q
    
    def collect_experience(self, experience):
        self.experience_replay.append(experience)
        return
    
    def sample_from_experience(self, sample_size):
        if(len(self.experience_replay) < sample_size):
            sample_size = len(self.experience_replay)   
        sample = random.sample(self.experience_replay, sample_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).float()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor([exp[3] for exp in sample]).float()   
        return s, a, rn, sn
    
    def train(self, batch_size ):
        s, a, rn, sn = self.sample_from_experience( sample_size = batch_size)
        if(self.network_sync_counter == self.network_sync_freq):
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0
        
        # predict expected return of current state using main network
        qp = self.q_net(s)
        pred_return, _ = torch.max(qp, axis=1)
        
        # get target return using target network
        q_next = self.get_q_next(sn)
        target_return = rn + self.gamma * q_next
        
        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        self.network_sync_counter += 1       
        return loss.item()

class Deep_Q():
    def main(self, gym, exp, cart, gamma, alpha, epsilon=1, iterations=1e6, replay_buffer_size=256):
        print_per_n = 1000
        env = gym.make('CartPole-v0')
        exp_replay_size = 256
        agent = DQN_Agent(seed= 1423, layer_sizes = [4, 64, 2], lr = alpha, sync_freq = 5, exp_replay_size=replay_buffer_size)

        # initiliaze experience replay      
        index = 0
        for i in range(replay_buffer_size):
            obs = env.reset()
            done = False
            while not done:
                A = agent.get_action(obs, env.action_space.n, epsilon=1)
                obs_next, reward, done, _ = env.step(A.item())
                agent.collect_experience([obs, A.item(), reward, obs_next])
                obs = obs_next
                index += 1
                if( index > replay_buffer_size):
                    break
                    
        # Main training loop
        losses_list, reward_list, episode_len_list, epsilon_list  = [], [], [], []
        index = 128
        epsilon = 1

        total_timesteps = 0
        total_timesteps_n_episodes = 0
        for i in range(int(iterations)):
            t = 0
            obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0
            while not done:
                ep_len += 1 
                A = agent.get_action(obs, env.action_space.n, epsilon)
                obs_next, reward, done, _ = env.step(A.item())
                agent.collect_experience([obs, A.item(), reward, obs_next])
            
                obs = obs_next
                rew  += reward
                index += 1
                t += 1
                total_timesteps += 1
                total_timesteps_n_episodes += 1
                if(index > 128):
                    index = 0
                    for j in range(4):
                        loss = agent.train(batch_size=16)
                        losses += loss      
            if epsilon > 0.05 :
                epsilon -= (1 / 5000)
            
            if i % print_per_n == 0:
                avg_timesteps = total_timesteps / (i+1) 

                print("Episode {} finished after {} timesteps".format(i, t+1))
                print("Average timesteps {}".format(avg_timesteps))
                print("Average timesteps of last {} episodes: {}\n".format(print_per_n, total_timesteps_n_episodes / print_per_n))
                
                exp.Episode_time(i, avg_timesteps, (total_timesteps_n_episodes / print_per_n))
                # exp.Episode_time(i, (total_timesteps_n_episodes / print_per_n))
                
                total_timesteps_n_episodes = 0

            losses_list.append(losses/ep_len), reward_list.append(rew), episode_len_list.append(ep_len), epsilon_list.append(epsilon)
            
        # exp.Loss_reward(losses_list, reward_list)
        
        return exp.df, losses_list, reward_list
        
        

class Tabular_Q():
    def main(self, gym, exp, cart, gamma, alpha, epsilon, zeroes, iterations=1e6):
          
        # Initialise the matrix with ones and zeroes
        if zeroes:
            Q = np.zeros([cart.STATE_SIZE, cart.ACTION_SIZE])
        else:
            Q = np.random.choice([0, 1], size=(cart.STATE_SIZE,cart.ACTION_SIZE), p=[.5,.5])

        print("Total matrix size: {}x{}".format(cart.STATE_SIZE, cart.ACTION_SIZE))
        print("Estimated GB used: {}GB".format(Q.nbytes/1000000000))
        print_per_n = 1000

        env = gym.make('CartPole-v0')
        env.reset()

        # observation --> 4-tuple: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
        # reward --> always 1: for every step the pole does not fall
        # done --> boolean: becomes true if the pole is >12degrees
        # info --> for debugging

        total_timesteps = 0
        total_timesteps_n_episodes = 0
        for episode in range(int(iterations)):
            done = False
            t = 0

            # Initialize state
            state = env.reset()
            state_index = cart.discretize(state)
            
            # We are done when the pole angle >= 12 degrees or we solved the problem
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample() # Explore state space
                else:
                    action = np.argmax(Q[state_index]) # Exploit learned values

                next_state, reward, done, info = env.step(action) # invoke Gym
                next_state_index = cart.discretize(next_state)
                next_max = np.max(Q[next_state_index])
                old_value = Q[state_index, action]
                new_value = old_value + alpha * (reward + gamma * next_max - old_value)
                Q[state_index, action] = new_value
                state = next_state
                state_index = next_state_index
                t += 1
                total_timesteps += 1
                total_timesteps_n_episodes += 1

            # Print statistics
            if episode % print_per_n == 0:
                avg_timesteps = total_timesteps / (episode+1) 

                print("Episode {} finished after {} timesteps".format(episode, t+1))
                print("Average timesteps {}".format(avg_timesteps))
                print("Average timesteps of last {} episodes: {}\n".format(print_per_n, total_timesteps_n_episodes / print_per_n))
           
                exp.Episode_time(episode, avg_timesteps, (total_timesteps_n_episodes / print_per_n))
                # exp.Episode_time(episode, (total_timesteps_n_episodes / print_per_n))

                total_timesteps_n_episodes = 0

        env.close()
        return exp.df
