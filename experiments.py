import pandas as pd
import matplotlib.pyplot as plt
import datetime

class Experiment_episode_timesteps:
  # This class is used to take care of recording results and adding them
  # to a dataframe.
  
  def __init__(self, _columns):
    self.df = pd.DataFrame(columns = _columns)

  def Episode_time(self, episode, avg_timesteps, avg_timesteps_last):
    """Simple function to create a dataframe entry to record episode and time

    Args:
        episode (int): current episode
        avg_timesteps (double): time steps averaged overall
        avg_timesteps_last (double): time steps of last 1000 episodes
    """    
    new_line = {}
    new_line["episodes"] = episode        
    new_line["avg_timesteps"] = avg_timesteps
    new_line["avg_timesteps_last"] = avg_timesteps_last

    self.df = self.df.append(new_line, ignore_index=True)

  def Save_df(self, df, file):
    # Function to save a dataframe to a csv file
    name = './csv/' + file + '.csv'
    df.to_csv(name, index = False, header=True)


  def Loss_reward(self, loss, reward, filename):
    # Function to print the loss and reward graph for Deep Q Learning
    df = pd.DataFrame({'losses':loss, 'reward':reward})

    # Add episodes
    df.insert(0, 'episodes', range(0, len(df)))
    
    fig, ax1 = plt.subplots()

    # plt.suptitle('Reward and Loss function', fontsize=14)


    color = 'tab:red'
    ax1.set_xlabel('Number of episodes')
    ax1.set_ylabel('Number of timesteps', color=color)
    ax1.plot(reward, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax1.set_title('Reward and Loss function')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(loss, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    

    plt.savefig('plots/{0}-{1}.png'.format(filename, datetime.datetime.now().strftime("%H:%M:%S")))

    plt.show()

  def Clear_df(self):
    # Function to empty a dataframe
    self.df = self.df[0:0]

  def Create_line_plot(self, df, filename, _title):
    """Simple function that creates a line plot of the given dataframe.

    Args:
        df (pd df): dataframe with TrueSkill scores of the bots
        filename (string): filename to be given
    """

    ax = df.plot.line(title=_title, x='episodes')
    
    ax.set_xlabel("Number of episodes")
    ax.set_ylabel("Number of timesteps")
    
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    plt.xlim([0, df['episodes'].max()])
    # plt.ylim([0, trueskill_max])

    # To make X axis nice integers
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.savefig('plots/{0}-{1}.png'.format(filename, datetime.datetime.now().strftime("%H:%M:%S")))
    plt.show()
