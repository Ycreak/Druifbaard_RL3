import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls


# Hyperparameters
buffer_size = 128 # Size of the batch to learn from
test_per_n = 5    # Size of testing sequence
steps = 20        # Maximum amount of learning steps before calling quits
max_score = 200   # Maximum reward value for one episode
_gamma = 0.99     # Gamma value for algorithm
_lr = 7e-3        # Learning rate of the agent optimizers
units = 128       # Neurons in the actor and critic layer
tf.random.set_seed(336_699)

# Critic network
class critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(units, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        v = self.v(x)
        return v
    
# Actor network
class actor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(units, activation='relu')
        self.a = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, input_data):
        x = self.d1(input_data)
        a = self.a(x)
        return a

class _agent():
    def __init__(self, gamma=0.99, clip_pram=0.2):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=_lr)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=_lr)
        self.actor = actor()
        self.critic = critic()
        self.clip_pram = clip_pram
        self.experience_replay = []
          
    def act(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])
  
    def collect_experience(self, experience):
        self.experience_replay.append(experience)

    def clear_buffer(self):
        self.experience_replay = []
    
    def actor_loss(self, probs, actions, adv, old_probs, closs):
        probability = probs      
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability, tf.math.log(probability))))
        sur1, sur2 = [], []
        
        for pb, t, op in zip(probability, adv, old_probs):
            t = tf.constant(t)
            op = tf.constant(op)
            ratio = tf.math.divide(pb, op)
            s1 = tf.math.multiply(ratio, t)
            s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram), t)
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        return loss

    def train(self, states, actions, adv, old_probs, discnt_rewards):
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




class PPO():
    def test_reward(self, env, agent):
        total_reward = 0
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(agent.actor(np.array([state])).numpy())
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        return total_reward

    def preprocess(self, states, actions, rewards, done, values, gamma, dones):
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

    def main(self, gym, exp, cart, gamma, alpha, iterations, _epochs=10, clip_pram=0.2):
        # Arguments
        _gamma = gamma
        _lr = alpha
        epochs = _epochs
        steps = iterations

        # Create the cartpole environment
        env = gym.make("CartPole-v0")

        # Create the agent
        agent = _agent(_gamma, clip_pram=clip_pram)

        target = False
        ep_reward, total_avgr, avg_rewards_list = [], [], []
        gross_reward, gross_n, total_episodes = 0, 0, 0
        actor_loss, critic_loss = [], []
        # Proceed in training steps
        for step in range(40):
            if target == True:
                break

            done = False
            state = env.reset()

            values, dones = [], []

            agent.clear_buffer()

            # Collect experience
            total_episodes += buffer_size
            for _ in range(buffer_size):
                action = agent.act(state)
                value = agent.critic(np.array([state])).numpy()
                next_state, reward, done, _ = env.step(action)
                prob = agent.actor(np.array([state]))
                dones.append(1 - done)
                values.append(value[0][0])

                # Add to buffer
                agent.collect_experience([state, action, reward, prob[0]])
                
                # Proceed to next state or reset environment
                state = next_state
                if done:
                    env.reset()
            
            # Pre-process experience
            value = agent.critic(np.array([state])).numpy()
            values.append(value[0][0])
            probs = [x[3] for x in agent.experience_replay]
            np.reshape(probs, (len(probs), 2))
            probs = np.stack(probs, axis=0)

            states, actions, returns, adv = self.preprocess([x[0] for x in agent.experience_replay], 
                                                            [x[1] for x in agent.experience_replay], 
                                                            [x[2] for x in agent.experience_replay],
                                                            dones, values, gamma, dones)

            # Train agent
            aloss, closs = 0, 0
            for _ in range(epochs):
                al, cl = agent.train(states, actions, adv, probs, returns)
                aloss += al.numpy()
                closs += cl.numpy()

            # Testing and statistics
            total_reward_per_n = 0
            for _ in range(test_per_n):
                rew = self.test_reward(env, agent)
                total_reward_per_n += rew
                gross_reward += rew
                gross_n += 1
            
            avg_reward = gross_reward / gross_n
            avg_reward_per_n = total_reward_per_n / test_per_n

            print("Testing sequence after {} episodes:".format(total_episodes))
            print("Average timesteps (reward) {}".format(avg_reward))
            print("Average timesteps (reward) of last {} episodes: {}\n".format(test_per_n, avg_reward_per_n))

            avg_rewards_list.append(avg_reward_per_n)
            actor_loss.append(aloss/epochs)
            critic_loss.append(closs/epochs)

            exp.Episode_time(total_episodes, avg_reward, avg_reward_per_n)


            if avg_reward_per_n == max_score or step >= steps:
                target = True

            env.reset()

        env.close()
        print(actor_loss)
        print(critic_loss)
        return exp.df, actor_loss, critic_loss, avg_rewards_list