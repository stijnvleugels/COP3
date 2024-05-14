import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class PenguinClusters:

    def __init__(self, num_particles:int, num_dim:int, rho:float, optimal_temp:float) -> None:
        self.num_particles = num_particles
        self.num_dim = num_dim
        self.rho = rho
        self.optimal_temp = optimal_temp
        self.boxsize = np.sqrt(num_particles/rho)

    def next_timestep(self, pos:np.ndarray, timestep:float) -> np.ndarray:
        ''' Calculate the next position of the particles given the current position.
            - `pos` must be (num_particles, ndim) of positions
            Returns the new positions of the particles and the particle field values at the new positions.
        '''
        gamma = 1
        Dt = 3e-5

        force, particle_field = self.force_field(pos)
        noise = np.random.normal(loc=0, scale=np.sqrt(2*timestep*Dt), size=pos.shape)
        new_pos = (pos + timestep / gamma * force + noise) % self.boxsize

        return new_pos, particle_field

    def force_field(self, pos:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ''' Calculate the force on each particle given the current position of all particles.
            Returns array of shape (num_particles, ndim) for the forces and the particles' field values array of shape (num_particles,).
        '''
        A = 1; kappa = 1 ; lamb = 1 
        force = np.zeros((pos.shape[0], pos.shape[1]))
        nearest_dists, diff_vectors = self.nearest_dist(pos)

        nearest_dists_inf = np.copy(nearest_dists) # has zeroes for the diagonal (should be in numerator)
        nearest_dists_inf[nearest_dists == 0] = np.inf # has infs for the diagonal (should be in denominator and exponents)
        
        for dim in range(force.shape[1]):
            force[:,dim] = - lamb * (np.sum(A*np.exp(-kappa*nearest_dists_inf)/nearest_dists_inf, axis=1) - self.optimal_temp) * \
                            np.sum( -A*np.exp(-kappa*nearest_dists_inf)*(kappa*nearest_dists + 1)*diff_vectors[:,:,dim] / nearest_dists_inf**3 , axis=1)
        particle_field = np.sum(A*np.exp(-kappa*nearest_dists_inf)/nearest_dists_inf, axis=1)
        return force, particle_field

    def nearest_dist(self, pos:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ''' Calculate the distance between all particles both as vector and length.
            Returns two arrays:
            - nearest_dists is (num_particles, num_particles) and contains the length of the distance vector between all particles
            - diff_vectors is (num_particles, num_particles, ndim) and contains the distance vector between all particle positions
        '''
        nearest_dists = np.zeros((pos.shape[0], pos.shape[0]))
        diff_vectors = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]))

        for particle in range(pos.shape[0]):
            diff_vectors[particle,:,:] = pos[particle,:] - pos[:,:]

        nearest_dists[:,:] = np.linalg.norm(diff_vectors[:,:,:], axis=2)
        return nearest_dists, diff_vectors
    
    def run_simulation(self, num_steps:int, timestep:float) -> tuple[np.ndarray, np.ndarray]:
        ''' Run the simulation for a number of steps and return the positions and particle field values at each timestep.
        '''
        total_positions = np.zeros([num_steps, self.num_particles, self.num_dim])
        particle_fields = np.zeros([num_steps, self.num_particles])
        positions = np.random.rand(self.num_particles, self.num_dim) * self.boxsize
        for i in tqdm(range(num_steps)):
            positions, particle_field = self.next_timestep(positions, timestep=timestep)
            total_positions[i, :] = positions
            particle_fields[i, :] = particle_field

        return total_positions, particle_fields
    
def plot_positions_and_fields(total_positions:np.ndarray, particle_fields:np.ndarray, optimal_temp:float):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(total_positions[-1,:,0], total_positions[-1,:,1], c=particle_fields[-1,:] - optimal_temp, cmap='viridis', alpha=0.5, s=10)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'))
    cbar.set_label('T - T$_{opt}$')
    plt.show()

def main():
    num_steps = 10**4
    rho = 1.3

    penguin_sim = PenguinClusters(num_particles=256, num_dim=2, rho=rho, optimal_temp=6)
    total_positions, particle_fields = penguin_sim.run_simulation(num_steps=num_steps, timestep=1e-4)

    plot_positions_and_fields(total_positions, particle_fields, optimal_temp=penguin_sim.optimal_temp)

    #TODO: when is the system in equilibrium? (plot the average particle field over time)

if __name__ == '__main__':
    main()