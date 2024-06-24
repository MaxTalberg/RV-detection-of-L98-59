import pickle
import numpy as np
import pandas as pd


def clean_and_pickle(espresso_path: str, harps_path: str, pickle_path: str):
    """
    Cleans the data from the ESPRESSO and HARPS instruments and pickles the results.

    Parameters
    ----------
    espresso_path : str
        File path for the ESPRESSO data file.
    harps_path : str
        File path for the HARPS data file.
    pickle_path : str
        Destination path to save the pickled data.

    Notes
    -----
    The function adjusts time values, filters out specific time points and handles
    missing values according to those specifified in D. S. Demangeon et al. (2021).
    The cleaned data is then saved into a pickle file format for persistence.
    """

    # --- HARPS
    # Define column titles
    column_titles = [
        "Time",
        "RV",
        "e_RV",
        "Halpha",
        "e_Halpha",
        "Hbeta",
        "e_Hbeta",
        "Hgamma",
        "e_Hgamma",
        "NaD",
        "e_NaD",
        "Sindex",
        "e_Sindex",
        "FWHM",
        "BIS",
    ]

    try:
        # Load HARPS data
        harps_df = pd.read_csv(harps_path, delim_whitespace=True, names=column_titles)
        harps_df["Time"] += 2457000  # Adjust the time column to BJD

        # Filter out specific BJDs from HARPS data
        excluded_bjds = [2458503.795048, 2458509.552019, 2458511.568314, 2458512.581045]
        cleaned_harps_df = harps_df[~harps_df["Time"].isin(excluded_bjds)].copy()

        # Replace invalid FWHM and BIS values with NaN
        cleaned_harps_df["FWHM"] = cleaned_harps_df["FWHM"].astype(str)
        cleaned_harps_df["FWHM"].replace("---", np.nan, inplace=True)
        cleaned_harps_df["FWHM"] = pd.to_numeric(
            cleaned_harps_df["FWHM"], errors="coerce"
        )
        cleaned_harps_df["BIS"].replace("---", np.nan, inplace=True)

    except Exception as e:
        print(f"Error processing HARPS data: {e}")

    # --- ESPRESSO
    # Define column titles
    espresso_column_titles = [
        "Time",
        "RV",
        "e_RV",
        "FWHM",
        "e_FWHM",
        "BIS",
        "e_BIS",
        "Contrast",
        "e_Contrast",
        "Sindex",
        "e_Sindex",
        "Halpha",
        "e_Halpha",
        "NaD",
        "e_NaD",
        "BERV",
        "Inst",
    ]

    try:
        # Load ESPRESSO data
        espresso_df = pd.read_csv(
            espresso_path, delim_whitespace=True, names=espresso_column_titles
        )
        espresso_df["Time"] += 2400000  # Adjust the time column to BJD

        # Filter out specific BJDs from ESPRESSO data with a tolerance
        excluded_bjds = [2458645.496, 2458924.639, 2458924.645]
        tolerance = 1e-3
        cleaned_espresso_df = espresso_df.copy()
        cleaned_espresso_df = cleaned_espresso_df[
            ~cleaned_espresso_df["Time"].apply(
                lambda x: any(abs(x - bjd) < tolerance for bjd in excluded_bjds)
            )
        ]

        # Split the data into pre and post fiber change
        cleaned_pre_df = cleaned_espresso_df[cleaned_espresso_df["Inst"] == "Pre"]
        cleaned_post_df = cleaned_espresso_df[cleaned_espresso_df["Inst"] == "Post"]

    except Exception as e:
        print(f"Error processing ESPRESSO data: {e}")

    # --- Organise data for pickling
    data_dict = {
        "ESPRESSO_pre": cleaned_pre_df,
        "ESPRESSO_post": cleaned_post_df,
        "HARPS": cleaned_harps_df,
    }

    try:
        # Save to pickle file
        with open(pickle_path, "wb") as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Error saving data to pickle: {e}")


def load_data_from_pickle(filepath):
    """
    Loads data from a pickle file.

    Parameters
    ----------
    filepath : str
        Path to the pickle file to be loaded.

    Returns
    -------
    data : dict
        A dictionary containing the data loaded from the pickle file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    EOFError
        If the file is empty or improperly formatted, indicating end of file reached without any data.
    Exception
        For other issues that might occur during the loading process.
    """
    try:
        # Open the pickle file in binary read mode
        with open(filepath, "rb") as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filepath} was not found.")
    except EOFError:
        raise EOFError(
            f"No data found in file {filepath}. The file may be corrupted or empty."
        )
    except Exception as e:
        raise Exception(f"An error occurred while loading the pickle file: {e}")
