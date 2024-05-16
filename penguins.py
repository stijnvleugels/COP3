import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

class PenguinClusters:

    def __init__(self, num_particles:int, num_dim:int, rho:float, optimal_temp:float) -> None:
        self.num_particles = num_particles
        self.num_dim = num_dim
        self.rho = rho
        self.optimal_temp = optimal_temp
        self.boxsize = np.sqrt(num_particles/rho)

    def init_positions(self) -> np.ndarray:
        ''' Initialize the positions of the particles in the box in four clusters.
        '''
        cluster1 = np.random.rand(self.num_particles//4, self.num_dim) * self.boxsize / 8 + np.array([self.boxsize/4, self.boxsize/4])
        cluster2 = np.random.rand(self.num_particles//4, self.num_dim) * self.boxsize / 8 + np.array([self.boxsize/4, self.boxsize/4]) + np.array([self.boxsize/2, self.boxsize/2])
        cluster3 = np.random.rand(self.num_particles//4, self.num_dim) * self.boxsize / 8 + np.array([self.boxsize/4, self.boxsize/4]) + np.array([self.boxsize/2, 0])
        cluster4 = np.random.rand(self.num_particles//4, self.num_dim) * self.boxsize / 8 + np.array([self.boxsize/4, self.boxsize/4]) + np.array([0, self.boxsize/2])
        return np.concatenate([cluster1, cluster2, cluster3, cluster4])

    def next_timestep(self, pos:np.ndarray, timestep:float) -> np.ndarray:
        ''' Calculate the next position of the particles given the current position.
            - `pos` must be (num_particles, ndim) of positions
            Returns the new positions of the particles and the particle field values at the new positions.
        '''
        gamma = 1
        Dt = 3e-5
        wind_strength = 0

        temp_force, particle_field, diff_vectors = self.force_field(pos)
        # covering_factor = self.total_covered_fraction(pos, diff_vectors)

        # wind_force = np.zeros(temp_force.shape)
        # wind_force[:,0] = (1-covering_factor) * wind_strength
        wind_force = np.zeros(temp_force.shape)

        noise = np.random.normal(loc=0, scale=np.sqrt(2*timestep*Dt), size=pos.shape)

        new_pos = (pos + timestep / gamma * (temp_force + wind_force) + noise) % self.boxsize

        return new_pos, particle_field

    def force_field(self, pos:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    
    def equilibrate(self, pos:np.ndarray, timestep:float) -> tuple[np.ndarray, np.ndarray]:
        ''' Equilibrate the system by running the simulation until the slope has sufficiently flattened and return the positions and particle field values at the end.
        '''
        num_steps = 10**5
        slope_check_interval = num_steps // 100
        particle_fields = np.zeros([slope_check_interval])
        old_slope = 0

        for i in range(num_steps):
            pos, particle_field = self.next_timestep(pos, timestep=timestep)
            particle_fields[i % slope_check_interval] = np.mean(particle_field)
            if i % slope_check_interval == 0:
                new_slope = np.median(np.abs(np.diff(particle_fields)))
                if np.abs(new_slope - old_slope) < 1e-4 and i > 0:
                    print(f'Equilibrated at timestep {i}, slope is {new_slope}')
                    return pos
                particle_fields = np.zeros([slope_check_interval])
                old_slope = new_slope
        print('Equilibration not reached, returning last positions')
        return pos

    def run_simulation(self, num_steps:int, timestep:float) -> tuple[np.ndarray, np.ndarray]:
        ''' Run the simulation for a number of steps and return the positions and particle field values at each timestep.
        '''
        total_positions = np.zeros([num_steps, self.num_particles, self.num_dim])
        particle_fields = np.zeros([num_steps, self.num_particles])
        positions = np.random.rand(self.num_particles, self.num_dim) * self.boxsize
        # positions = self.init_positions()
        # positions = self.equilibrate(positions, timestep=timestep)

        for i in tqdm(range(num_steps)):
            positions, particle_field = self.next_timestep(positions, timestep=timestep)
            total_positions[i, :] = positions
            particle_fields[i, :] = particle_field

        return total_positions, particle_fields
    
    
def plot_positions_and_fields(total_positions:np.ndarray, particle_fields:np.ndarray, optimal_temp:float, filename:str, Tjit:bool=False) -> None:
    def update(frame):
        sc.set_offsets(total_positions[frame,:,0:2])
        if not Tjit:
            sc.set_array(particle_fields[frame,:] - optimal_temp)
        text.set_text(f'Time: {frame}')
        return sc, text,

    # plot positions as function of time in a video
    fig, ax = plt.subplots(figsize=(5,5))
    if not Tjit:
        sc = ax.scatter(total_positions[0,:,0], total_positions[0,:,1], c= particle_fields[0,:] - optimal_temp, cmap='viridis', s=10, vmin=-1, vmax=1)
    else:
        sc = ax.scatter(total_positions[0,:,0], total_positions[0,:,1], c=optimal_temp, cmap='viridis', s=10)
    text = ax.text(s='', x=0.5, y=1.05, ha='center', va='center', transform=ax.transAxes)
    cbar = plt.colorbar(sc)
    if not Tjit:
        cbar.set_label('T - T$_{opt}$')
    else:
        cbar.set_label('T$_{opt}$')
    frames_to_show = np.arange(0, total_positions.shape[0], 100)
    ani = FuncAnimation(fig, update, frames=frames_to_show)
    ani.save(f'{filename}.gif', writer='pillow', fps=5)
    # plt.show()

def plot_field_time(particle_fields:np.ndarray, optimal_temp:float, filename:str) -> None:
    fig, ax = plt.subplots()
    ax.plot(np.mean( np.abs(particle_fields), axis=1) - optimal_temp)
    ax.set_xlabel('Time')
    ax.set_ylabel('<|T - T$_{opt}$|>')
    plt.savefig(f'{filename}.png')
    plt.clf()

def main():
    num_steps = 1*10**4
    timestep = 5e-5

    # with jitter in the optimal temperature
    for temperature in [20]:
        temperature_jit = temperature + np.concatenate([np.ones(100)*0.0, np.ones(50)*0.4])*temperature
        penguin_sim = PenguinClusters(num_particles=150, num_dim=2, rho=1, optimal_temp=temperature_jit)
        total_positions, particle_fields = penguin_sim.run_simulation(num_steps=num_steps, timestep=timestep)
        plot_positions_and_fields(total_positions, particle_fields, penguin_sim.optimal_temp, f'penguin_simulation_{temperature}_jitterT', Tjit=True)

    # just the optimal temperature
    for temperature in [20]:
        penguin_sim = PenguinClusters(num_particles=150, num_dim=2, rho=1, optimal_temp=temperature)
        total_positions, particle_fields = penguin_sim.run_simulation(num_steps=num_steps, timestep=timestep)
        plot_positions_and_fields(total_positions, particle_fields, penguin_sim.optimal_temp, f'penguin_simulation_{temperature}')
        plot_field_time(particle_fields, penguin_sim.optimal_temp, f'penguin_field_{temperature}')

    #TODO: 
    # - compare properties vs method of initialisation
    # - is it ok that equilibrium is now done by checking slope flattening (i.e. second derivative)? slope is still always nonzero so couldn't look at flattening of the slope itself(?)
    # - write results to file so we can do the plots later

if __name__ == '__main__':
    main()