# Binary Parameter Fitting

This is a Python code for fitting the parameters of a binary star system by using the Emcee MCMC algorithm. The code takes some known parameters of the system and uses them to generate a light curve of the system. Then it adds some noise to the data and uses Emcee to fit the parameters of the system based on the generated data.
## Status

**Note:** This code is currently a work-in-progress and may not be fully functional. The documentation will be updated as the project progresses.

**Note:** This code was written prior to the release of PHOEBE 2.3. 

### Usage
* Install the necessary packages: phoebe, emcee, matplotlib, scipy, schwimmbad, corner.
* Import the necessary packages and define the known parameters of the binary system.
* Generate a light curve of the system with noise added.
* Define the initial distribution, priors, and functions necessary for the MCMC fitting.
* Run the MCMC algorithm to fit the parameters of the binary system.
* Generate various plots of the MCMC results to visualize the fitted parameters and their uncertainties.

### Example

An example usage of the code with the given parameters is shown in the script. The code generates a light curve of a binary system, adds some noise to the data, and then uses Emcee to fit the parameters of the system. The MCMC results are plotted in various figures to visualize the fitted parameters and their uncertainties.
