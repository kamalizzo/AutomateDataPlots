# AutomateDataPlots
 Process raw data and generate plots for thesis purpose.
 Run the main.py to process electrochemistry data or run Jupyter Notebook file for vibration measurement.

Current Implementation

Electrochemistry features are:
1) Generate raw data in xlsx file
2) Generate measurement and I-U curves (polarization curves) plots
3) Generate EIS spectra plots (Nyquist and Bode) 
4) Generate table consisting appropriate gas flow and water vapor flow based on current measuring point, relative humidity and stoichiometry.


Vibration feature is:
5) Process and measure raw data of vibration (acceleration) from ESP32 into Time-Domain analysis and FFT/PSD analysis

Packages/dependencies:
1) pandas
2) matplotlib
3) numpy
4) scipy
