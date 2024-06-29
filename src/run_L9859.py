from l9859_model import L9859Analysis


def main():
    """
    Main function to run nested sampling Gaussian Process (GP) regression model
    on the L98-59 system, using radial velocity (RV) data from HARPS and ESPRESSO.

    It configures the analysis based on specified parameters, including the
    inclusion of specific dataset features and algorithmic settings.

    Raises
    ------
    Exception
        General exception raised during execution of the analysis.
    """
    filepath = "datasets/cleaned_data_20240531.pickle"
    include_planet_b = False
    include_fwhm = False
    include_sindex = False
    algorithm_params = {
        "do_clustering": True,
        "precision_criterion": 1e-9,
        "num_repeats": 5,
        "read_resume": False,
        "nprior": 5000,
        "nfail": 50000,
    }

    output_params = {
        "base_dir": "output_dir/final/specific_run_b_{}_fwhm_{}_sindex_{}".format(
            include_planet_b, include_fwhm, include_sindex
        ),
        "feedback": 1,
    }

    try:
        analysis = L9859Analysis(
            filepath,
            include_planet_b,
            include_fwhm,
            include_sindex,
            algorithm_params,
            output_params,
        )
        analysis.run_analysis()
        analysis.handle_results()
    except Exception as e:
        print(f"An error occurred during the analysis: {e}")


if __name__ == "__main__":
    main()
