from src.DataProcessing import PolCurveData, EISData
from src.GenerateData import GenerateData
from src.GenerateMeasurementPoints import GasFlowCalc, HumidityCalc, \
    MeasurementPoints


if __name__ == "__main__":
    # Generate Plots and Data Process
    data = PolCurveData("C:/Users/kamal/Thesis-ZBT/datas/01_VV_datas")
    gen_data = GenerateData(data)
    gen_data.generate_all()

    # Generate Measuring Points for Measurements at Lab 6
    currents = [1, 2, 5, 7, 10, 13, 15, 17, 20, 25, 30]
    stoichiometries = [(2, 5), (1.75, 5), (1.5, 5)]
    humidities = [(100, 100), (75, 100)]
    mp = MeasurementPoints(currents, stoichiometries, humidities)
    mp.generate_table()

