import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
from typing import Union
import os
import functools
import matplotlib as mpl

def set_defaults():
    ''' Set default values for matplotlib rcParams. '''
    # set default font size for axis labels
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12

    # set default legend font size
    mpl.rcParams['legend.fontsize'] = 12

def CatchFileNotFoundError(function):
    ''' Catch FileNotFoundError and create the directory if it doesn't exist;
        if the error is not caused by a missing directory, raise the error '''

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except FileNotFoundError as e:
            path = str(e).split("'")[1]
            if os.path.isfile(path):
                raise e
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                result = function(*args, **kwargs)
        return result
    return wrapper

@CatchFileNotFoundError
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
    ani.save(f'gifs/{filename}.gif', writer='pillow', fps=5)
    # plt.show()

@CatchFileNotFoundError
def plot_field_time(particle_fields:np.ndarray, optimal_temp:float, filename:str) -> None:
    fig, ax = plt.subplots()
    ax.plot(np.mean(np.abs(particle_fields - optimal_temp), axis=1))
    ax.set_xlabel('Time')
    ax.set_ylabel('<|T - T$_{opt}$|>')
    ax.set_yscale('log')
    plt.savefig(f'figures/{filename}.png')
    plt.clf()

def read_from_pickle(filename:str) -> tuple[np.ndarray, np.ndarray, Union[float,np.ndarray], float]:
    ''' Reads the data from the pickle file given filename, including extension. Returns the positions, particle fields, optimal temperature, density and timestep size.'''
    try:
        simulation_data = pickle.load(open(f'model_output/{filename}', 'rb'))
    except FileNotFoundError:
        raise FileNotFoundError(f'No data found for model_output/{filename}. Run the simulation first.')
    return simulation_data['positions'], simulation_data['particle_fields'], simulation_data['optimal_temp'], simulation_data['rho'], simulation_data['timestep']

def main():
    set_defaults()

    # for filename, Tjit in zip(['test', 'test_jitterT'],[False, True]):
    for filename in os.listdir('model_output'):
        total_positions, particle_fields, optimal_temp, rho, timestep = read_from_pickle(f'{filename}')
        plot_positions_and_fields(total_positions, particle_fields, optimal_temp, filename[:-2], Tjit=False)
        plot_field_time(particle_fields, optimal_temp, filename[:-2])

if __name__ == '__main__':
    main()