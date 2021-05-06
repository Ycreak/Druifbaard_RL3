
class Random_actions:

    # def __init__(self, gym, exp, iterations):
    #     pass

    def main(self, gym, exp, iterations):
    
        env = gym.make('CartPole-v0')
        env.reset()

        total_timesteps = 0
        for episode in range(int(iterations)):
            done = False
            t = 0
            
            # Initialize state
            state = env.reset()

            while not done:
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                state = next_state
                t = t + 1
                total_timesteps = total_timesteps + 1
            # Print statistics
            if episode % 1000 == 0:
                avg_timesteps = total_timesteps / (episode+1) 
                
                print("Episode {} finished after {} timesteps".format(episode, t+1))
                print("Average timesteps {}\n".format(avg_timesteps))
            
                exp.Episode_time(episode, avg_timesteps, 0)
                # exp.Episode_time(episode, avg_timesteps)




        env.close()
        
        return exp.df

