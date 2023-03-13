# Reproducability
This project was developed in Ubuntu 22.04, with Python 3.10.6.

1. Create a new virtual environment with: `python -m venv <name of environment>`
2. Load your environment.
3. Install dependencies with: `pip install -r requirements.txt`
4. Make two folders, `data/songs/` and `data/annotations/`. 
   1. In the first one, place the albums from the Ballroom dataset.
   2. In the second one, place the annotations, with the .beats extension
5. Use main to generate the csv files from the respective similarity functions. If you need to change the function, change the similarity function on line 36 of main.py to `mwmd`, `mse` or `spectral_flux` and change the name of the csv file to be generated on line 24.
6. In order to generate the plots, create a folder plots and run `generate_heatmaps.py` or `generate_relative_plotsl.py`. If you need to change the similarity function, change the string in line 5 to `mse`, `mwmd`, `spectral flux`.

## Structure
This project is based on 5 different python scripts/modules.

1. data_parser.py: A script with a class to read, transform and serve audio files with their respective annotations.
2. distance_functions.py: A package with functions that implement the similarity functions to be tested.
3. dsp_functions.py: A module with functions for the dsp functionalities used by the beat tracking system, except for the filters
4. filters.py: a module with a mean and median filter.
5. main.py: the main of this project to process and generate a csv with per song metrics.
