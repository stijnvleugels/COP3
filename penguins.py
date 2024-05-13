import numpy as np
import matplotlib.pyplot as plt

def next_timestep(pos:np.ndarray, timestep:float, boxsize:float) -> np.ndarray:
    ''' Calculate the next position and velocity of the particles given the current position.
        - `pos` must be (num_particles, ndim) of positions
    '''
    gamma = 1
    Dt = 3e-5

    force = force_field(pos, boxsize)[0]
    noise = np.random.normal(loc=0, scale=np.sqrt(2*timestep*Dt), size=pos.shape)

    new_pos = (pos + timestep / gamma * force + noise) % boxsize

    return new_pos

def force_field(pos:np.ndarray, boxsize:float) -> tuple[np.ndarray, np.ndarray, float]:
    ''' Calculate the force on each particle given the current position of all particles.
        Returns array of shape (num_particles, ndim) for the forces and nearest_dists of shape (num_particles, num_particles).
    '''
    A = 1; kappa = 1 ; lamb = 1 ; optimal_temp = 6
    force = np.zeros((pos.shape[0], pos.shape[1]))
    nearest_dists, diff_vectors = nearest_dist(pos, boxsize)

    nearest_dists_inf = np.copy(nearest_dists) # has zeroes for the diagonal (should be in exponent; I would think inf would also work for that but for some reason it doesn't)
    nearest_dists_inf[nearest_dists == 0] = np.inf # has infs for the diagonal (should be in denominator)
    
    for dim in range(force.shape[1]):
        force[:,dim] = - lamb * (np.sum(A*np.exp(-kappa*nearest_dists)/nearest_dists_inf, axis=1) - optimal_temp) * \
                        np.sum( -A*np.exp(-kappa*nearest_dists)*(kappa*nearest_dists + 1)*diff_vectors[:,:,dim] / nearest_dists_inf**3 , axis=1)
    return force, nearest_dists

def nearest_dist(pos, boxsize) -> tuple[np.ndarray, np.ndarray]:
    ''' Calculate the distance between all particles both as vector and length.
        Returns two arrays:
        - nearest_dists is (num_particles, num_particles) and contains the length of the distance vector between all particles
        - diff_vectors is (num_particles, num_particles, ndim) and contains the distance vector between all particle positions
    '''
    nearest_dists = np.zeros((pos.shape[0], pos.shape[0]))
    diff_vectors = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]))

    for particle in range(pos.shape[0]):
        diff_vectors[particle,:,:] = (pos[particle,:] - pos[:,:] + boxsize/2) % boxsize - boxsize/2

    nearest_dists[:,:] = np.linalg.norm(diff_vectors[:,:,:], axis=2)
    return nearest_dists, diff_vectors

def main():
    timestep = 0.0001
    num_steps = 10000
    boxsize = 10
    num_particles = 50
    num_dim = 2

    positions = np.random.rand(num_particles, num_dim) * boxsize

    total_positions = np.zeros([num_steps, num_particles, num_dim])
    for i in range(num_steps):
        positions = next_timestep(positions, timestep=timestep, boxsize=boxsize)
        total_positions[i, :] = positions

    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(total_positions[:,:,0], total_positions[:,:,1], color='red', alpha=0.002)
    ax.scatter(total_positions[-1,:,0], total_positions[-1,:,1], color='blue')
    ax.set(xlim=(0,boxsize), ylim=(0,boxsize))
    plt.show()

if __name__ == '__main__':
    main()