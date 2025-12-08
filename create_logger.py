## File to define the function(s) used to control the logger section of the project


## Imports

# From this project (constants)
from utils.constants import PATH_LOGS_FOLDER, LOG_END_FILE

# From this project (functions)
# - None

# From sys library
import logging
import sys

# From external libraries
# - None


## Functions

def create_logger(logger_level_console: int, logger_level_file: int, log_file_name: str, libraries_to_ignore_logs_from: list[str]) -> None:
    """
    Function to create a general logger (both for console printing and file saving) that will work across all modules in a project

    To set up a logger at each module, each file should include the following line:
        logger = logging.getLogger(__name__)

    Parameters:
        logger_level_console (int): Level of logging for the console logger
        logger_level_file (int): Level of logging for the file logger
        log_file_name (str): name of the file to save the logs to
        libraries_to_ignore_logs_from (list[str]): list of the name of each library that might send internal info logs and we want to avoid logging in our logger

    Returns:
        None
    """

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # The lowest level to give freedom to console and file loggers

    # Only add handlers if this is the first time this is called (avoid duplicates)
    if not root_logger.hasHandlers():
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logger_level_console)
        console_handler.setFormatter(logging.Formatter('%(levelname)s | %(name)s | %(funcName)s | %(message)s'))

        # File handler
        file_handler = logging.FileHandler(PATH_LOGS_FOLDER + log_file_name + LOG_END_FILE, mode='w')
        file_handler.setLevel(logger_level_file)
        file_handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(funcName)s | %(message)s"))

        # Add handlers
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    # Silence info logs from external libraries
    for lib in libraries_to_ignore_logs_from:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    return None