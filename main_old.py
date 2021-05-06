import gym
import slimevolleygym

from collections import deque
import numpy as np
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

# Environment settings
EPISODES = 20000

# Exploration settings
epsilon = 1

##
PRINT_PER_N = 1

def action_int_to_binary(index):
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


class RandomPolicy:
  def __init__(self):
    self.action_space = gym.spaces.MultiBinary(3)
    pass
  def predict(self, obs):
    return self.action_space.sample()

class VanillaDQN:
  def build_model(self):
    hidden_size = 10

    model = Sequential()

    model.add(Dense(hidden_size, activation='relu', input_dim=12))
    model.add(Dense(hidden_size, activation='relu'))
    # model.add(Conv2D(256, (3, 3), input_shape=(84, 168, 3)))  # OBSERVATION_SPACE_VALUES = (84, 168, 3) a 84x168 RGB image.
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D())
    # model.add(Dropout(0.2)) # Prevents overfitting

    # model.add(Conv2D(256, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D())
    # model.add(Dropout(0.2))

    #model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    #model.add(Dense(64))

    model.add(Dense(8, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
    model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model

  def predict(self, state):
    return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

  def __init__(self):
    # Build the model, target model and equalize weights
    self.model = self.build_model()
    self.target_model = self.build_model()
    self.target_model.set_weights(self.model.get_weights())

    # Keep track of states/actions in the replay memory
    self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    # Update target network after n steps
    self.target_update_counter = 0

  # Trains main network every step during episode
  def train(self, terminal_state, step):
      if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
          return

      minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

      # Get current states from minibatch, then query NN model for Q values
      current_states = np.array([transition[0] for transition in minibatch])/255
      current_qs_list = self.model.predict(current_states)

      # Get future states from minibatch, then query NN model for Q values
      # When using target network, query it, otherwise main network should be queried
      new_current_states = np.array([transition[3] for transition in minibatch])/255
      future_qs_list = self.target_model.predict(new_current_states)

      X = []
      y = []

      # Now we need to enumerate our batches
      for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
          # If not a terminal state, get new q from future states, otherwise set it to 0
          # almost like with Q Learning, but we use just part of equation here
          if not done:
              max_future_q = np.max(future_qs_list[index])
              new_q = reward + DISCOUNT * max_future_q
          else:
              new_q = reward

          # Update Q value for given state
          current_qs = current_qs_list[index]
          current_qs[action] = new_q

          # And append to our training data
          X.append(current_state)
          y.append(current_qs)

      # Fit on all samples as one batch, log only on terminal state
      self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

      # Update target network counter every episode
      if terminal_state:
          self.target_update_counter += 1

      # If counter reaches set value, update target network with weights of main network
      if self.target_update_counter > UPDATE_TARGET_EVERY:
          self.target_model.set_weights(self.model.get_weights())
          self.target_update_counter = 0


env = gym.make("SlimeVolley-v0")
policy_dqn = VanillaDQN()
policy_random = RandomPolicy()

# obs = env.reset()
# done = False
# total_reward = 0

# while not done:
#   action = policy.predict(obs)
#   print(action)
#   exit()
#   obs, reward, done, info = env.step(action)
#   total_reward += reward
#   env.render()

reward_list, episode_len_list, epsilon_list  = [], [], []
total_steps, total_steps_n_episodes = 0, 0
total_reward, total_reward_n_episodes = 0, 0
for episode in range(EPISODES):
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    obs = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = action_int_to_binary(np.argmax(policy_dqn.predict(obs)))
        else:
            # Get random action
            action = policy_random.predict(obs)

        new_obs, reward, done, _ = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # Every step we update replay memory and train main network
        policy_dqn.replay_memory.append((obs, action, reward, new_obs, done))
        policy_dqn.train(done, step)

        obs = new_obs

        # For stats
        step += 1
        total_steps += 1
        total_steps_n_episodes += 1
        total_reward += reward
        total_reward_n_episodes += reward

    if epsilon > 0.05 :
      epsilon -= (1 / 5000)
    
    # Stats
    if episode % PRINT_PER_N == 0:
      avg_steps = total_steps / (episode+1) 
      avg_reward = total_reward / (episode+1)
      print("Average timesteps {} ---- (of last {} episodes: {}".format(avg_steps, PRINT_PER_N, total_steps_n_episodes / PRINT_PER_N))       
      print("Average reward {} ---- (of last {} episodes: {}".format(avg_reward, PRINT_PER_N, total_reward_n_episodes / PRINT_PER_N))          
      total_steps_n_episodes = 0
      total_reward_n_episodes = 0

      # Add reward

    reward_list.append(episode_reward), episode_len_list.append(step), epsilon_list.append(epsilon)