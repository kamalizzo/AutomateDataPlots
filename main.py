from src.electrochem.DataPolCurve import PolCurveData, PC1, PC2
from src.electrochem.DataEIS import EISData
from src.electrochem.GenerateData import GenerateData
from src.electrochem.GenerateMeasurementPoints import GasFlowCalc, HumidityCalc, \
    MeasurementPoints
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

    # Generate Plots and Data Process
    """Include the folder where the txt file of the measurement as the first argument/ kwarg fdir, 
        the chosen name for the document in the second arg/kwarg filename e.g:"""
    data_1 = PolCurveData(fdir="C:/Users/kamal/Thesis-ZBT/datas/12_60_3rd/datas", filename='PC_1')
    #
    # data_2 = EISData(fdir="C:/Users/kamal/Thesis-ZBT/datas/04_VV_datas/eis", filename='EIS_1')

    # Generate Data of the Plots
    """Include the object data_1 or data_2 inside the GenerateData class,
        use method generate_files(ftype=['pdf', 'png', 'xls']) to generate file separately for each plots for each 
        defined file type or generate_compiled_pdf() to compiled all plots in a single pdf"""
    # gen_data = GenerateData(data_1)
    # # if kwarg path/first argument is None, a generated folder will be included in the current working directory,
    # # where all files will be generated in the folder, if filetype is None, 'pdf' will be generated
    # gen_data.generate_files(filetype=['pdf', 'png'])  # or
    # gen_data.generate_compiled_pdf()   # or
    #
    # gen_data.generate_all() # to generate all pdf and png of the plots in data_1/data_2

    # Generate Measuring Points for Measurements at Lab 6
    """Measure mass flow (Nml/min) for air and hydrogen given the current points, stoichiometries and humidities"""
    # currents = [74.5, 49.5, 24.5]
    # stoichiometries = [(2, 2), (2, 5)]
    # humidities = [(75, 50), (75, 75), (75, 100)]
    # mp = MeasurementPoints(currents, stoichiometries, humidities)
    # mp.generate_table('Flowtabelle 1. Case 2nd')

