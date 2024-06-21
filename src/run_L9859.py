from l9859_model import L9859Analysis


def main():
    filepath = "datasets/cleaned_data_20240531.pickle"
    include_planet_b = False
    include_fwhm = True
    include_sindex = False
    algorithm_params = {
        "do_clustering": True,
        "precision_criterion": 1,
        "num_repeats": 1,
        "read_resume": False,
        "nprior": 500,
        "nfail": 5000,
    }

    output_params = {
        "base_dir": "output_dir/specific_run_b_{}_fwhm_{}_sindex_{}/".format(
            include_planet_b, include_fwhm, include_sindex
        ),
        "feedback": 1,
    }

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


if __name__ == "__main__":
    main()
