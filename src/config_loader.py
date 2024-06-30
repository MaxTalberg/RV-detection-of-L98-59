import configparser


def load_config(config_file):
    """
    Load a configuration file using the configparser library.

    Parameters
    ----------
    config_file : str
        The path to the configuration file that needs to be loaded.

    Returns
    -------
    configparser.ConfigParser
        A ConfigParser object containing the configuration settings from the specified file.
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    return config
