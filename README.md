# RealEstateMLProject

<img src="new_york.gif" width="250" height="250"/>

Predicting apartment prices in NewYork based on interesting characteristics. For example rating of schools in the area, parking, hotels, public transportation, and more… The general goal of the project is that given the address of an apartment in New York, we can easily calculate the price of the apartment, even for a person who does not understand real estate at all.

- [RealEstateMLProject](#realestatemlproject)
  * [Running The Project](#running-the-project)
    + [Running Remotely](#running-remotely)
  * [Installation Instructions](#installation-instructions)
    + [Libraries to Install](#libraries-to-install)
  * [References](#references)
  
## Running The Project
|Step      | Description |
|-------------|---------|
|1| Clone the project to your local computer |
|2| Install required packages in requiremnts.txt |


### Running Remotely
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/itayisraelov/RealEstateMLProject.git/master)

## Installation Instructions

1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at: https://www.anaconda.com/distribution/
2. Create a new environment for the Project:
In Windows open `Anaconda Prompt` from the start menu, in Mac/Linux open the terminal and run `conda create --name torch`. Full guide at https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
3. To activate the environment, open the terminal (or `Anaconda Prompt` in Windows) and run `conda activate torch`
4. Install the required libraries according to the table below (to search for a specific library and the corresponding command you can also look at https://anaconda.org/)

### Libraries to Install

|Library         | Command to Run |
|----------------|---------|
|`Jupyter Notebook`|  `conda install -c conda-forge notebook`|
|`numpy`|  `conda install -c conda-forge numpy`|
|`matplotlib`|  `conda install -c conda-forge matplotlib`|
|`pandas`|  `conda install -c conda-forge pandas`|
|`scipy`| `conda install -c anaconda scipy `|
|`scikit-learn`|  `conda install -c conda-forge scikit-learn`|
|`seaborn`|  `conda install -c conda-forge seaborn`|


## References

