import numpy as np
import tensorflow as tf 
import gym
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls

env= gym.make("CartPole-v0")


class critic(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(128,activation='relu')
    self.v = tf.keras.layers.Dense(1, activation = None)

  def call(self, input_data):
    x = self.d1(input_data)
    v = self.v(x)
    return v
    

class actor(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(128,activation='relu')
    self.a = tf.keras.layers.Dense(2,activation='softmax')

  def call(self, input_data):
    x = self.d1(input_data)
    a = self.a(x)
    return a

class agent():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma
        # self.a_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        # self.c_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.actor = actor()
        self.critic = critic()
        self.clip_pram = 0.2
        self.experience_replay = []
          
    def act(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])
  
    def collect_experience(self, experience):
        self.experience_replay.append(experience)

    def actor_loss(self, probs, actions, adv, old_probs, closs):
        probability = probs      
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        #print(probability)
        #print(entropy)
        sur1 = []
        sur2 = []
        
        for pb, t, op in zip(probability, adv, old_probs):
                        t =  tf.constant(t)
                        op =  tf.constant(op)
                        #print(f"t{t}")
                        #ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
                        ratio = tf.math.divide(pb,op)
                        #print(f"ratio{ratio}")
                        s1 = tf.math.multiply(ratio,t)
                        #print(f"s1{s1}")
                        s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram),t)
                        #print(f"s2{s2}")
                        sur1.append(s1)
                        sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        #closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        #print(loss)
        return loss

    def train(self, states, actions,  adv , old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p),2))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v =  self.critic(states,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)
            
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss

def test_reward(env):
  total_reward = 0
  state = env.reset()
  done = False
  while not done:
    action = np.argmax(agent.actor(np.array([state])).numpy())
    next_state, reward, done, _ = env.step(action)
    state = next_state
    total_reward += reward

  return total_reward

def preprocess1(states, actions, rewards, done, values, gamma):
    g = 0
    lmbda = 0.95
    returns = []
    for i in reversed(range(len(rewards))):
       delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
       g = delta + gamma * lmbda * dones[i] * g
       returns.append(g + values[i])

    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    returns = np.array(returns, dtype=np.float32)
    return states, actions, returns, adv    


tf.random.set_seed(336699)
agent = agent()
steps = 5000
ep_reward = []
total_avgr = []
target = False 
best_reward = 0
avg_rewards_list = []
buffer_size = 128
test_per_n = 5
epochs = 10

gross_reward = 0
gross_n = 0
total_episodes = 0
for s in range(steps):
  if target == True:
    break
  
  done = False
  state = env.reset()
  all_aloss = []
  all_closs = []
  rewards = []
  states = []
  actions = []
  probs = []
  dones = []
  values = []

# COLLECT EXP
  total_episodes += buffer_size
  for e in range(buffer_size):
   
    action = agent.act(state)
    value = agent.critic(np.array([state])).numpy()
    next_state, reward, done, _ = env.step(action)
    dones.append(1-done)
    rewards.append(reward)
    states.append(state)
    actions.append(action)
    prob = agent.actor(np.array([state]))
    probs.append(prob[0])
    values.append(value[0][0])
    state = next_state
    if done:
      env.reset()
  
# PREPROCESS EXP
  value = agent.critic(np.array([state])).numpy()
  values.append(value[0][0])
  np.reshape(probs, (len(probs),2))
  probs = np.stack(probs, axis=0)

  states, actions,returns, adv  = preprocess1(states, actions, rewards, dones, values, 1)

# TRAIN
  for _ in range(epochs):
      al,cl = agent.train(states, actions, adv, probs, returns)

  # Testing and statistics
  total_reward_per_n = 0
  for _ in range(test_per_n):
    rew = test_reward(env)
    total_reward_per_n += rew
    gross_reward += rew
    gross_n += 1
  
  avg_reward = gross_reward / gross_n
  avg_reward_per_n = total_reward_per_n / test_per_n

  print("Testing sequence after {} episodes:".format(total_episodes))
  print("Average timesteps (reward) {}".format(avg_reward))
  print("Average timesteps (reward) of last {} episodes: {}\n".format(test_per_n, avg_reward_per_n))

  avg_rewards_list.append(avg_reward_per_n)

  if avg_reward_per_n == 200:
    target = True

  env.reset()

env.close()