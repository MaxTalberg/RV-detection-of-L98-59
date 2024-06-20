import pickle
import numpy as np
import pandas as pd


def clean_and_pickle(espresso_path: str, harps_path: str, pickle_path: str):
    """
    Clean and pickle the data from the ESPRESSO and HARPS instruments.
    """

    # --- HARPS
    # Column titles
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

    # Load the data
    harps_df = pd.read_csv(harps_path, delim_whitespace=True, names=column_titles)

    # Adjust the time column to BJD by adding 2457000
    harps_df["Time"] += 2457000

    # Clean the data
    excluded_bjds = [2458503.795048, 2458509.552019, 2458511.568314, 2458512.581045]
    cleaned_harps_df = harps_df[~harps_df["Time"].isin(excluded_bjds)].copy()

    # Missing vals to nan
    cleaned_harps_df["FWHM"] = cleaned_harps_df["FWHM"].astype(str)
    cleaned_harps_df["FWHM"].replace("---", np.nan, inplace=True)
    cleaned_harps_df["FWHM"] = pd.to_numeric(cleaned_harps_df["FWHM"], errors="coerce")

    cleaned_harps_df["BIS"].replace("---", np.nan, inplace=True)

    # --- ESPRESSO
    # Column titles
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

    # Load the data
    espresso_df = pd.read_csv(
        espresso_path, delim_whitespace=True, names=espresso_column_titles
    )

    # Adjust the time column to BJD by adding 2400000
    espresso_df["Time"] += 2400000

    # Clean the data
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

    # --- Organise data for pickling
    data_dict = {
        "ESPRESSO_pre": cleaned_pre_df,
        "ESPRESSO_post": cleaned_post_df,
        "HARPS": cleaned_harps_df,
    }

    # Save to pickle
    with open(pickle_path, "wb") as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data_from_pickle(filepath):
    # Open the pickle file in binary read mode
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    return data
