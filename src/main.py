from config_loader import load_config
from pickle_data import clean_and_pickle
from periodograms import run_periodogram_plot


def main():
    # Initialise the configuration file
    config = load_config("config.ini")

    # Load the paths from the configuration file
    espresso_path = config["Paths"]["espresso"]
    harps_path = config["Paths"]["harps"]
    pickle_path = config["Paths"]["pickle"]

    # Clean and pickle the data
    clean_and_pickle(espresso_path, harps_path, pickle_path)

    # Run the periodogram plot
    run_periodogram_plot(pickle_path, ESPRESSO=True)
    run_periodogram_plot(pickle_path, ESPRESSO=False)


if __name__ == "__main__":
    main()
