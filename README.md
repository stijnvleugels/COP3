This set of python files allows you to run a 2D simulation of penguin clustering using Brownian motion and plot the results.

Run the simulation in the terminal with the following command: \
`python penguins.py [seed] [temperature] [num_particles] [savesampling] [num_steps] [timestep]` \
it is possible to skip the arguments at the end, they will be set to defaults, but it's impossible to skip an argument and then set the next one in this order.
The models are saved to \
- model_output/t{temperature}_d{rho}_n{num_particles}_equid_run{i} for the signle penguin class simulations (only one optimal temperature)
- model_output/t{temperature}_d{rho}_n{num_particles}_jitterT_equid_run{i} for the two penguin class simulations (two different optimal temperature groups)
Each simulation is run three separate times to study the effect of random initialisation and random motion on the results; hence run{i} wil range from 1 to 3.

Producing the plots is done using: \
`python plots.py [filename]`
This will write results to several folders, each created if they do not exist yet. Results for all three runs are automatically produced.  

The arguments have the following meaning: 
 - seed is the random seed (integer). 
 - temperature is the optimal temperature for the penguins to reach
 - num_particles is the number of penguins in the system
 - savesampling is the number of timesteps between each saved step; e.g., 100 means that only every 100th step is saved and written to a file
 - num_steps is the number of steps that will be taken after the system has equilibrated
 - timestep is the size of each timestep; the total simulated time is then t = timesteps*num_steps
 - filename is the filename up until the _run{i} component, i.e. 't{temperature}_d{rho}_n{num_particles}_jitterT_equid' and excluding extension.
Note that rho=1 by default.