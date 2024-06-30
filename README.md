# Re-examining the radial velocity detection of L 98-59 b
[![Documentation Status](https://readthedocs.org/projects/rv-detection-of-l98-59/badge/?version=latest)](https://rv-detection-of-l98-59.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Animation of the L98-59 system
![Three Planets Orbiting](animation/three_planets.gif)

#### Written by Max Talberg
##### email: mt942@cam.ac.uk
##### June, 2024
##### Updated: July, 2024

## __Introduction__

In recent years the radial velocity (RV) method has proven exceptionally effective at detecting
exoplanets. Advancements in instruments like HARPS (Mayor et al.
2003) and ESPRESSO (Pepe et al. 2021) have pushed the detection threshold to new lows.
Introducing L98-59b, a terrestrial planet with half the mass of
Venus, the lowest mass planet detected using RVs to date (Demangeon et al. 2021).
We aim to re-examine the detection of L98-59b, utilising a Gaussian Process (GP) framework.
This is the Data Analysis Pipeline for the project.


## Run the L98-59 Model Locally
The Data Analysis Pipeline contains the code to preprocess and produce periodograms from the HARPS and ESPRESSO dataets, run the L98-59 model and generate results.

### Setting Up the Project

1. **Clone the repository:**
   - Shallow clone the repository from GitLab:
     ```bash
     git clone --depth 1 git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/projects/mt942.git
     ```

2. **Set up the virtual environment:**
   - Navigate to the project directory:
     ```bash
       cd mt942
       ```
   - Create virtual environment:
     ```bash
     conda env create -f environment.yml
     ```
    - Activate virtual environment:
      ```bash
      conda activate l9859-env
      ```
   - The L98-59 model requires PolyChordLite (Handley et al. (2015)) be installed first:
        ```bash
        git clone https://github.com/PolyChord/PolyChordLite.git
        cd PolyChordLite
        make
        pip install .
        cd ..
        ```
3. **Running the script:**

   - Running the main script to preprocess and produce periodograms from the HARPS and ESPRESSO datasets:
        ```bash
     python src/main.py
        ```
   - The L98-59 model is ran from, the `run_l9859.py` script where one can be tune the L98-59 model with different parameters:
        ```bash
        python src/run_l9859.py
        ```

## Running the script on Docker

### Setting Up the Project

1. **Clone the repository:**
   - Clone the repository from GitLab:
     ```bash
     git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/A1_SKA_Assessment/mt942.git
     ```

2. **Build the Docker image:**
    - Navigate to the project directory:
      ```bash
      cd mt942
      ```
   - Build image:
     ```bash
     docker build -t ska-project .
     ```

3. **Running the script:**

   - Run the main script:
     ```bash
     docker run -v host_directory:/app/plots ska-project
     ```
        - Replace `host_directory` with the path to the directory where you want to save the plots, for example: `/path/to/plots` and all the images will be saved into a folder named `plots`, information acompanying will be in the terminal output.


## Documentation
The documentation can be accessed [here](https://rv-detection-of-l98-59.readthedocs.io/en/latest/) or generated locally using the following steps:

### Generating Documentation
1. **Navigate to the `docs` directory:**

      ```bash
       cd docs
     ```
2. **Generate the documentation:**

      ```bash
       make html
     ```
3. **Open the documentation:**

      ```bash
       open build/html/index.html

## Testing

### Running Unit Tests

1. **Navigate to the project directory `mt942 and run Unit Tests:**

      ```bash
       pytest
     ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Use of generative tools

This project has utilised auto-generative tools in the development of documentation that is compatible with auto-documentation tools,
latex formatting and the development of plotting functions.

Example prompts used for this project:
- Generate doc-strings in NumPy format for this function.
- Generate Latex code for a subplot.
- Generate Latex code for a 3 by 3 matrix.
- Generate Python code for a 2 by 1 subplot.
