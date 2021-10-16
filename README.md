# On mass conservation for predicting the long term water balance of the rainfall runoff process
Conservation of mass is a fundamental law of hydrological science. It has been proposed that conservation laws might not be benefitial to hydrological models, so long as those models can incorporate the bias between mass in (usually precipitation mass) and mass out (usually runoff mass, but also including mass loss to groundwater infiltration and evapotranspiration). We use deep learning, physics-informed deep learning,conceptual and process-based hydrology models to test the role of conservation laws in streamflow predictions. We find that both deep learning models (pure data-driven and physics-informed) have a better long term mass balance representation than the conceptual and process-based hydrology models. Explicit mass balance conservation mechanism is not required for deep learning models to achieve long term mass balance.

# Results
## results_analysis.ipynb
This is the script that calculates all the metrics. The calculation of cummulative mass takes quite a lot of time.
## paper_table.ipynb
This is the script to make the tables with the standard time series metrics.
## mass_balance.ipynb
This is the script to make the figures based on the cummulative mass balance.
## results_analysis_test_ensemble.ipynb
This script tests the sensitivity of the analysis to initial conditions (random seeds for training).
# Data
This directory just has the CAMELS attributes and the USGS gauge information that is used in the analysis. All the forcing data are hosted on the [main CAMELS website](https://ral.ucar.edu/solutions/products/camels), no need for us to host them again. The model results are hosted on CUAHSI Hydroshare (link coming soon).
# Figs
here are the figures.
