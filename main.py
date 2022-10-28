from src.DataPolCurve import PolCurveData, PC1, PC2
from src.DataEIS import EISData
from src.GenerateData import GenerateData
from src.GenerateMeasurementPoints import GasFlowCalc, HumidityCalc, \
    MeasurementPoints
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # Generate Plots and Data Process
    # # data = PolCurveData("C:/Users/kamal/Thesis-ZBT/datas/01_VV_datas")
    # data_c1 = PolCurveData("C:/Users/kamal/Thesis-ZBT/datas/04_VV_datas/datas",
    #                        'Case 3 [80⁰C]')
    # # data_c1a = PolCurveData("C:/Users/kamal/Thesis-ZBT/datas/09/datas",
    # #                         'Case 3 [80⁰C]')
    data_c1 = PolCurveData("C:/Users/kamal/Thesis-ZBT/datas/12_60_3rd/datas",
                           'Case 1 [60⁰C](June)')
    # # data_c2 = PolCurveData("C:/Users/kamal/Thesis-ZBT/datas/05/datas",
    # #                        'Case 1 [60⁰C]')
    #
    # # data_c3 = PolCurveData('C:/Users/kamal/Thesis-ZBT/datas/10/datas',
    # #                        'Case 3 [80⁰C](21. Mai)')
    # # data_c3 = PolCurveData("C:/Users/kamal/Thesis-ZBT/datas/06_datas/datas",
    # #                        'Case 2 [70⁰C]')
    # # data_c3a = PolCurveData("C:/Users/kamal/Thesis-ZBT/datas/08/datas",
    # #                         'Case 2 [70⁰C]')
    # # print(data.dict_chosen_df['1'])

    # data_3a = EISData("C:/Users/kamal/Thesis-ZBT/datas/04_VV_datas/eis",
    #                   'Case 3 [80⁰C] (1st)')

    # df = pd.read_csv("C:/Users/kamal/Thesis-ZBT/datas/12_60 "
    #                  "3rd/datas/12_20220611.txt", encoding='ISO-8859-1',
    #                  sep='\t', decimal=',')
    # print(df)
    # data_1a = EISData("C:/Users/kamal/Thesis-ZBT/datas/10_80 3rd/eis",
    #                   'Case 3 [80⁰C] (1st)')
    # data_1a.gamry_to_zplot()
    # print(data_1a.fdir)
    # tf = df



    # data.generate_nyqs(xrange=(0.0005, 0.045), yrange=(0.001, -0.0175))
    # data.generate_bods(yrange=(0.0005, 0.03))

    # gen_data = GenerateData(data_c1a)
    # # gen_data.generate_raw_data(gen_data.chosen_df)
    # # gen_data.generate_compiled_pdf()
    # gen_data.generate_all()
    # gen_data.generate_all_raw_data()
    # for key in data_c1.generated_figures.keys():
    #     if 'Measurement Plots' in key:
    #         hey = data_c1.generated_figures[key]['fig']
    # plt.show()#
    # gen_data = GenerateData(data_c1a)
    # gen_data2 = GenerateData(data_c1)
    # gen_data2.generate_compiled_pdf()

    # gen_data2.generate_compiled_pdf('80_new')
    # print(data_c3a.dict_chosen_df)
    # print(data_c3.dict_chosen_df)
    print(data_c1.dataframe)






    # Generate Measuring Points for Measurements at Lab 6
    # currents = [74.5, 49.5, 24.5]
    # stoichiometries = [(2, 2), (2, 5)]
    # humidities = [(75, 50), (75, 75), (75, 100)]
    # mp = MeasurementPoints(currents, stoichiometries, humidities)
    # mp.generate_table('Flowtabelle 1. Case 2nd')

