#DruifBaard_RL2

To install dependencies, run the following command:

	$ pip3 install -r requirements.txt

To run the various implementations, use the following:
	$ python3 main.py random		(to use a random playing agent)
	$ python3 main.py tabular		(to use Tabular Q-learning)
	$ python3 main.py deep			(to use Deep Q-learning)
	$ python3 main.py mcpg			(to use Monte Carlo Policy Gradient)
	
To run the various experiments, use the following:
	$ python3 main.py exp_rnd_tab	(to compare random and tabular)
	$ python3 main.py exp_zeroes	(to compare the initialisation of the numpy array)
	$ python3 main.py exp_large		(to compare two sizes of tables)
	$ python3 main.py exp_tab_e		(to compare epsilon values for tabular)
	$ python3 main.py exp_tab_a		(to compare alpha values for tabular)
	$ python3 main.py exp_tab_g		(to compare gamma values for tabular)
	$ python3 main.py exp_deep		(to plot the results of Deep Q)
	$ python3 main.py exp_mcpg		(to plot the results of MCPG)
	$ python3 main.py exp_all		(to compare deep and MCPG)

	$ python3 main.py exp_deep_e	(to compare epsilon values for deep)