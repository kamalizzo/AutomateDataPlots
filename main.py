from src.DataPolCurve import PolCurveData, PC1, PC2
from src.DataEIS import EISData
from src.GenerateData import GenerateData
from src.GenerateMeasurementPoints import GasFlowCalc, HumidityCalc, \
    MeasurementPoints


if __name__ == "__main__":
    # Generate Plots and Data Process
    # data = PolCurveData("C:/Users/kamal/Thesis-ZBT/datas/01_VV_datas")
    data = PolCurveData("C:/Users/kamal/Thesis-ZBT/datas/04_VV_datas/datas",
                        'chosen')

    # data = EISData("C:/Users/kamal/Thesis-ZBT/datas/04_VV_datas/eis")

    gen_data = GenerateData(data)
    # gen_data.generate_raw_data(gen_data.chosen_df)
    gen_data.generate_compiled_pdf()



    # Generate Measuring Points for Measurements at Lab 6
    # currents = [74.5, 49.5, 24.5]
    # stoichiometries = [(2, 2), (2, 5)]
    # humidities = [(75, 50), (75, 75), (75, 100)]
    # mp = MeasurementPoints(currents, stoichiometries, humidities)
    # mp.generate_table('Flowtabelle 1. Case 2nd')

