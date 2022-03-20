from src.DataProcessing import PolCurveData, EISData
from src.GenerateData import GenerateData


if __name__ == "__main__":
    data = PolCurveData("C:/Users/kamal/Thesis-ZBT/datas/01_VV_datas")
    gen_data = GenerateData(data)
    # gen_data.generate_compiled_pdf()
    # gen_data.generate_files()
    # gen_data.generate_raw_data()
    gen_data.generate_all()
