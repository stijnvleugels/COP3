import numpy as np
import matplotlib.pyplot as plt

def next_timestep(pos:np.ndarray, vel:np.ndarray, timestep:float, boxsize:float) -> tuple[np.ndarray, np.ndarray, np.array, np.array, float]:
    ''' Calculate the next position and velocity of the particles given the current position and velocity.
        - `pos` must be (num_particles, ndim) of positions, 
        - `vel` must be (num_particles, ndim) of velocities
    '''
    old_force = force(pos, boxsize)[0]
    new_pos = (pos + timestep * vel + timestep**2 / 2 * old_force) % boxsize

    new_force, new_nearest_dists = force(new_pos, boxsize)
    new_vel = vel + timestep / 2 * (new_force + old_force)

    return new_pos, new_vel, new_nearest_dists

def force(pos:np.ndarray, boxsize:float) -> tuple[np.ndarray, np.ndarray, float]:
    ''' Calculate the force on each particle given the current position of all particles.
        Returns array of shape (num_particles, ndim) for the forces, nearest_dists of shape (num_particles, num_particles) and the pressure.
    '''
    kappa = 1 ; lamb = 1 ; Top = 1
    force = np.zeros((pos.shape[0], pos.shape[1]))
    nearest_dists, diff_vectors = nearest_dist(pos, boxsize)

    nearest_dists_inf = np.copy(nearest_dists) # has zeroes for the diagonal (should be in exponent; I would think inf would also work for that but for some reason it doesn't)
    nearest_dists_inf[nearest_dists == 0] = np.inf # has infs for the diagonal (should be in denominator)
    
    for dim in range(force.shape[1]):
        force[:,dim] = - lamb * (np.sum(np.exp(-kappa*nearest_dists)/nearest_dists_inf, axis=1) - Top) * \
                        np.sum( - np.exp(-kappa*nearest_dists)*(kappa*nearest_dists + 1)*diff_vectors[:,:,dim] / nearest_dists_inf**3 , axis=1)
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
    positions = np.random.rand(10, 2) * 10
    velocities = np.random.rand(10, 2) * 10
    timestep = 0.001
    boxsize = 10

    for i in range(10000):
        positions, velocities = next_timestep(positions, velocities, timestep=timestep, boxsize=boxsize)[0:2]

    fig, ax = plt.subplots()
    ax.scatter(positions[:,0], positions[:,1], color='blue')
    plt.show()

if __name__ == '__main__':
    main()