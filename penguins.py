import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import pickle
import os
from plots import CatchFileNotFoundError
import sys

''' 
This script can be used to simulate the behaviour of a group of penguins (particles) in a 2D box with Brownian motion, that aim to optimize their temperature by clustering together.
The output is a pickle file containing the positions and particle field values at each timestep, which can be used to visualize the simulation with plots.py.
'''

class PenguinClusters:

    def __init__(self, num_particles:int, num_dim:int, rho:float, optimal_temp:float, results_filename:str, wind_strength:float=0) -> None:
        self.num_particles = num_particles
        self.num_dim = num_dim
        self.rho = rho
        self.optimal_temp = optimal_temp
        self.wind_strength = wind_strength
        self.boxsize = np.sqrt(num_particles/rho)
        self.results_filename = results_filename

    def cluster_init(self) -> np.ndarray:
        ''' Initialize the positions of the particles in the box in four clusters spaced half a boxsize apart in each x, y.
        '''
        cluster1 = np.random.rand(self.num_particles//4, self.num_dim) * self.boxsize / 8 + np.array([self.boxsize/4, self.boxsize/4])
        cluster2 = np.random.rand(self.num_particles//4, self.num_dim) * self.boxsize / 8 + np.array([self.boxsize/4, self.boxsize/4]) + np.array([self.boxsize/2, self.boxsize/2])
        cluster3 = np.random.rand(self.num_particles//4, self.num_dim) * self.boxsize / 8 + np.array([self.boxsize/4, self.boxsize/4]) + np.array([self.boxsize/2, 0])
        cluster4 = np.random.rand(self.num_particles//4, self.num_dim) * self.boxsize / 8 + np.array([self.boxsize/4, self.boxsize/4]) + np.array([0, self.boxsize/2])
        return np.concatenate([cluster1, cluster2, cluster3, cluster4])

    def next_timestep(self, pos:np.ndarray, timestep:float) -> tuple[np.ndarray, np.ndarray]:
        ''' Calculate the next position of the particles given the current position.
            - `pos` must be (num_particles, ndim) of positions
            - `timestep` is the time step size
            Returns the new positions of the particles and the particle field values at the new positions.
        '''
        gamma = 1
        Dt = 3e-5
        temp_force, particle_field, diff_vectors = self.force_field(pos)

        if self.wind_strength != 0: 
            covering_factor = self.total_covered_fraction(pos, diff_vectors)
            wind_force = np.zeros(temp_force.shape)
            wind_force[:,0] = (1-covering_factor) * self.wind_strength
        else:
            wind_force = np.zeros(temp_force.shape)

        noise = np.random.normal(loc=0, scale=np.sqrt(2*timestep*Dt), size=pos.shape)
        new_pos = (pos + timestep / gamma * (temp_force + wind_force) + noise) % self.boxsize

        return new_pos, particle_field

    def force_field(self, pos:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Calculate the force on each particle given the current position of all particles.
            Returns array of shape (num_particles, ndim) for the forces and, particles' field values array of shape (num_particles,) and the difference vectors.
        '''
        A = 1; kappa = 1 ; lamb = 1 
        force = np.zeros((pos.shape[0], pos.shape[1]))
        dists, diff_vectors = self.particle_dists(pos)

        dists_inf = np.copy(dists) # has zeroes for the diagonal (should be in numerator)
        dists_inf[dists == 0] = np.inf # has infs for the diagonal (should be in denominator and exponents)
        
        for dim in range(force.shape[1]):
            force[:,dim] = - lamb * (np.sum(A*np.exp(-kappa*dists_inf)/dists_inf, axis=1) - self.optimal_temp) * \
                            np.sum( -A*np.exp(-kappa*dists_inf)*(kappa*dists + 1)*diff_vectors[:,:,dim] / dists_inf**3 , axis=1) 
        particle_field = np.sum(A*np.exp(-kappa*dists_inf)/dists_inf, axis=1) 
        return force, particle_field, diff_vectors

    def covered_fraction(self, upwind_particles:np.ndarray, particle_radius:float) -> tuple[float, float]:
        ''' Find the particle that covers the most and compute the coverage fraction of this particle.
            Returns coverage fraction and the corresponding most covering value.
        '''
        most_covering_idx = np.argmin(np.abs(upwind_particles[:,1]))
        most_covering_value = upwind_particles[most_covering_idx,1]
        fraction = ( 2*particle_radius-np.abs(most_covering_value) ) / (2*particle_radius)
        return fraction, most_covering_value

    def total_covered_fraction(self, pos:np.ndarray, diff_vectors:np.ndarray) -> np.ndarray:
        ''' Compute the total coverage fraction of all particles. To this end the upwind particles are masked (those that are positioned left
            of the considered particle, where their radius is taken into account), then the particle that most covers the given particle is taken
            and its covering fraction is computed, after which the second most covering particle, at the other side of the wind stream, is used to
            get to the total coverage fraction.
            Returns array of shape (num_particles) for the covered fraction per particle
        '''
        particle_radius = 0.2
        fraction = np.zeros((pos.shape[0]))
        for particle in range(diff_vectors.shape[0]):
            upwind_mask = (diff_vectors[particle,:,0] > 0) & (diff_vectors[particle,:,1] < 2*particle_radius) & (diff_vectors[particle,:,1] > -2*particle_radius)
            upwind_particles = diff_vectors[particle, upwind_mask]

            if (len(upwind_particles!=0)):
                fraction[particle], most_covering_value = self.covered_fraction(upwind_particles, particle_radius)

                if (most_covering_value == 0):
                    continue
                elif (most_covering_value > 0):
                    other_upwind_particles = upwind_particles[(upwind_particles[:,1] < 0)] 
                elif (most_covering_value < 0):
                    other_upwind_particles = upwind_particles[(upwind_particles[:,1] > 0)]

                if (len(other_upwind_particles!=0)):
                    fraction[particle] += self.covered_fraction(other_upwind_particles, particle_radius)[0]
                    if fraction[particle] > 1:
                        fraction[particle] = 1
        return fraction

    def particle_dists(self, pos:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ''' Calculate the distance between all particles both as vector and length.
            Returns two arrays:
            - dists is (num_particles, num_particles) and contains the length of the distance vector between all particles
            - diff_vectors is (num_particles, num_particles, ndim) and contains the distance vector between all particle positions
        '''
        dists = np.zeros((pos.shape[0], pos.shape[0]))
        diff_vectors = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]))

        for particle in range(pos.shape[0]):
            diff_vectors[particle,:,:] = pos[particle,:] - pos[:,:]

        dists[:,:] = np.linalg.norm(diff_vectors[:,:,:], axis=2)
        return dists, diff_vectors
    
    def equilibrate(self, pos:np.ndarray, timestep:float) -> tuple[np.ndarray, np.ndarray]:
        ''' Equilibrate the system by running the simulation until the slope has sufficiently flattened and return the positions and particle field values at the end.
        '''
        num_steps = 10**5
        slope_check_interval = num_steps // 25 # check only 4% of the steps
        particle_fields = np.zeros([slope_check_interval])
        slope_threshold = 1e-5 * np.min(self.optimal_temp)/3 # higher temperature, weaker threshold
        old_slope = 0

        print('Equilibrating...')
        for i in range(num_steps):
            pos, particle_field = self.next_timestep(pos, timestep=timestep)
            particle_fields[i % slope_check_interval] = np.mean(particle_field)
            if i % slope_check_interval == 0:
                new_slope = np.median(np.abs(np.diff(particle_fields)))
                if np.abs(new_slope - old_slope) < slope_threshold and i > 0:
                    print(f'Equilibrated at timestep {i}')
                    return pos
                particle_fields = np.zeros([slope_check_interval])
                old_slope = new_slope
        print('Equilibrium not reached, returning last positions')
        
        return pos
    
    @CatchFileNotFoundError
    def write_results_to_file(self, pos:np.ndarray, particle_fields:np.ndarray, timestep:float, filename:str, savesampling:int) -> None:
        ''' Write the results to a pickle file. '''
        pos_samp = pos[::savesampling]
        particle_fields_samp = particle_fields[::savesampling]
        pickle.dump({'positions':pos_samp, 'particle_fields':particle_fields_samp, 'savesampling':savesampling,
                     'optimal_temp':self.optimal_temp, 'rho':self.rho, 'timestep':timestep},
                    open(f'model_output/{filename}.p', 'wb'))

    def run_simulation(self, num_steps:int, timestep:float, savesampling:int=100) -> tuple[np.ndarray, np.ndarray]:
        ''' Run the simulation for a number of steps and return the positions and particle field values at each timestep.
        '''
        total_positions = np.zeros([num_steps, self.num_particles, self.num_dim])
        particle_fields = np.zeros([num_steps, self.num_particles])
        positions = np.random.rand(self.num_particles, self.num_dim) * self.boxsize
        positions = self.equilibrate(positions, timestep=timestep)

        for i in tqdm(range(num_steps)):
            positions, particle_field = self.next_timestep(positions, timestep=timestep)
            total_positions[i, :] = positions
            particle_fields[i, :] = particle_field

        self.write_results_to_file(total_positions, particle_fields, timestep, self.results_filename, savesampling)
        return total_positions, particle_fields

def main():
    args = [341, 10, 256, 100, 10000, 0.0001] # default values
    rho = 1
    for i, arg in enumerate(sys.argv):
        if i > 0: # first arg is this file's name
            args[i-1] = arg

    seed, temperature, num_particles, savesampling, num_steps, timestep = args
    np.random.seed(int(seed))
    temperature = float(temperature)
    num_particles = int(num_particles)
    num_steps = int(num_steps)
    timestep = float(timestep)
    savesampling = int(savesampling)
    
    # use optimal temperature directly
    for i in range(3):
        penguin_sim = PenguinClusters(num_particles=num_particles, num_dim=2, rho=rho, optimal_temp=temperature, results_filename=f't{temperature}_d{rho}_n{num_particles}_equid_run{i+1}')
        total_positions, particle_fields = penguin_sim.run_simulation(num_steps=num_steps, timestep=timestep, savesampling=100)

    # with jitter in the optimal temperature; two classes of penguins - the 'babies' and the 'adults'
        
    normal_particles_num = num_particles * 3//4
    jitter_particles_num = num_particles - normal_particles_num
    temperature_jit = temperature + np.concatenate([np.ones(normal_particles_num)*0.0, np.ones(jitter_particles_num)*0.4])*temperature
    for i in range(3):
        penguin_sim = PenguinClusters(num_particles=num_particles, num_dim=2, rho=rho, optimal_temp=temperature_jit, results_filename=f't{temperature}_d{rho}_n{num_particles}_jitterT_equid_run{i+1}')
        total_positions, particle_fields = penguin_sim.run_simulation(num_steps=num_steps, timestep=timestep, savesampling=savesampling)

if __name__ == '__main__':
    main()