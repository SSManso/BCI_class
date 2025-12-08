## File to define the function(s) used to control the plotting section of the project

## Imports

# From this project (constants)
from utils.variable_constants import SAVE_PLOTS, LOG_FILE_NAME
from utils.constants import PATH_PLOTS_FOLDER, PLOT_END_FILE

# From this project (functions)
# - None

# From sys library
import logging
import os

# From external libraries
import matplotlib.pyplot as plt
import mne
import numpy as np


## Set up logger
logger = logging.getLogger(__name__)


## Define path to save and format type
PATH_SAVE = PATH_PLOTS_FOLDER + LOG_FILE_NAME + '/'
FORMAT_TYPE = PLOT_END_FILE.split('.')[1]

## Functions

def check_and_create_folder(path: 'str') -> None:
    '''
    Function to check if a folder (used to save plots) exists, and if it doesn't, create it
    
    Parameters:
        path (str): path to check
    
    Returns:
        None
    '''

    if not os.path.exists(path):
        os.makedirs(path)

    return None

def tmp_plot(tmp: np.ndarray) -> None:
    """
    Function to create a tmp plot and save it

    This will only happen if the global variable SAVE_PLOTS is True

    Parameters:
        tmp (np.ndarray): vector to plot 

    Returns:
        None
    """

    if SAVE_PLOTS:
        name_plot = 'tmp'

        plt.figure()

        plt.plot(tmp)

        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.grid(True)

        plt.tight_layout()
        check_and_create_folder(PATH_SAVE)
        plt.savefig(PATH_SAVE + name_plot + PLOT_END_FILE, format=FORMAT_TYPE)
        plt.close()

        logger.info(f'Plot {name_plot} saved')

    return None