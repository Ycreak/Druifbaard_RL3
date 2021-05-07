# Reinforcement Learning 3
Philippe Bors
Job van der Zwaag
Luuk Nolden

To install dependencies, run the following command:

	$ pip3 install -r requirements.txt

To run the latest implementation, use the following:
	$ python3 main.py ppo			(to use Proximal Policy Optimization)
	
To run the various experiments, use the following:
	$ python3 main.py exp_ppo				(to create a plot of PPO's performance)
	$ python3 main.py exp_ppo_loss			(to create a plot of the loss/reward function)
	$ python3 main.py exp_ppo_gamma			(to create a plot of tweaking gamma)
	$ python3 main.py exp_ppo_alpha			(to create a plot of tweaking alpha)
	$ python3 main.py exp_ppo_epochs		(to create a plot of tweaking epochs)		
	$ python3 main.py exp_ppo_epsilon		(to create a plot of tweaking epsilon)				
	$ python3 main.py exp_ppo_tweaked		(to create a plot of a completely tweaked PPO)				

To pitch MCPG against PPO, two functions need to be run:
	$ python3 main.py exp_ppo_tweaked
	$ python3 main.py exp_mcpg_tweaked

These save their dataframes to a file on the disk, which we combined using LibreCalc to create the plot seen in the paper.