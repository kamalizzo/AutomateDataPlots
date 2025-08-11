import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from typing import Mapping

from abc import ABC, abstractmethod, ABCMeta


from src.plot.DataProcessing import ECData, Plot, DFType, Datas, Datasets, DataType


class AccelerationData(ECData, ABC):
    def __new__(cls, fdir, filename=None, data_source=None, **kwargs):
        return super(AccelerationData, cls).__new__(AC2) if data_source == "Lab" else super(AccelerationData, cls).__new__(AC1)

    def __init__(self, fdir, filename=None, **kwargs):
        super().__init__(fdir, filename, **kwargs)

    @classmethod
    def _set_datatype(cls):
        return 'vib'


class AC1(AccelerationData):
    def __init__(self, fdir, filename=None, **kwargs):
        super().__init__(fdir, filename=filename, **kwargs)

    def set_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.fdir, sep=';', decimal='.', names=['ind', 'time', 'x', 'y', 'z'], header=None)
        df.columns = df.iloc[0]
        df = df[1:]
        return df 
    
    def set_datas(self):
        datas: Datas = {
            DataType.accel: {'x': self.dataframe['t'].astype(float) / 1000000, 
                            'y': {'accX': self.dataframe['AccX'].astype(float),
                                    'accY': self.dataframe['AccY'].astype(float),
                                    'accZ': self.dataframe['AccZ'].astype(float) - 1.0}}}
                                   

        return datas

    def set_datasets(self):
        return super().set_datasets()
    
    def generate_list_curves(self):
        pass

    def generate_full_dict(self):
        pass

    def generate_chosen_dict(self):
        pass

    def generate_datafigure(self):
        pass



class AC2(AccelerationData):
    def __init__(self, fdir, filename=None, **kwargs):
        super().__init__(fdir, filename, **kwargs)

    def set_dataframe(self) -> Mapping[str, pd.DataFrame] :
        return pd.read_csv(self.fdir, header=None, names=['Hz', 'g^2/Hz', 'c3', 'c4', 'c5'],encoding='ISO-8859-1', sep='\t', decimal='.', skiprows=43)
    
    def set_datas(self):
        datas: Datas = {
            DataType.accel: {'x': self.dataframe['t'], 
                            'y': {'accX': self.dataframe['AccX'].astype(float),
                                    'accY': self.dataframe['AccY'].astype(float),
                                    'accZ': self.dataframe['AccZ'].astype(float) - 1.0}},
            DataType.psd: {'x': self.dataframe['t'].astype(float) / 1000000, 
                           'y': {'accX': self.dataframe['AccX'].astype(float),
                                    'accY': self.dataframe['AccY'].astype(float),
                                    'accZ': self.dataframe['AccZ'].astype(float) - 1.0}}
            }
        return datas

    def set_datasets(self):
        pass
        # self.datasets: Datasets = {'x': self.dataframe[''], 
        #                            'y': {'accX': self.dataframe['AccX'].astype(float),
        #                                  'accY': self.dataframe['AccY'].astype(float),
        #                                  'accZ': self.dataframe['AccZ'].astype(float) - 1.0},
        #                            'type': DataType.accel}
    
    def generate_list_curves(self):
        return super().generate_list_curves()
    
    def generate_full_dict(self):
        return super().generate_full_dict()
    
    def generate_chosen_dict(self):
        dict_chosen = {}

        return super().generate_chosen_dict()
    
    def generate_datafigure(self):
        return super().generate_datafigure()        
    
