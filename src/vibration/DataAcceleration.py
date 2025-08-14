import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from typing import Mapping, cast

from abc import ABC, abstractmethod, ABCMeta

from src.plot.DataProcessing import ECData, Plot, DFType, Datas, Datasets, DataType, YDataSetsAccel

from scipy.signal import welch



class AccelerationData(ECData, ABC):
    def __new__(cls, fdir, filename=None, data_source=None, **kwargs):
        return super(AccelerationData, cls).__new__(AC2) if data_source == "Lab" else super(AccelerationData, cls).__new__(AC1)

    def __init__(self, fdir, filename=None, **kwargs):
        super().__init__(fdir, filename, **kwargs)

    @classmethod
    def _set_datatype(cls):
        return 'vib'

    @staticmethod
    def __set_data_dict(datas: Datas):
        time_data = datas['datas'][DataType.accel]['x']
        accel_data = datas['datas'][DataType.accel]['y']
        psd_data = datas['datas'][DataType.psd]

        ad_v: YDataSetsAccel
        for ad_k, ad_v in accel_data:
            accel = ad_v.get("data")
            mean = round(np.mean(accel), 5)
            rms = round(np.sqrt(np.mean(accel ** 2)), 2)
            peak = round(np.max(np.abs(accel)), 2)
            avg = 5
            npsg = len(accel) / avg
            fs = 4096
            ff_welch, Pxxx_welch = welch(accel, fs=fs, nperseg=npsg, scaling='density', window='hann')

            peak_idx = np.argmax(Pxxx_welch)
            df = ff_welch[peak_idx]
            peak_value = Pxxx_welch[peak_idx]

            half_max = peak_value / 2.0
            above_half = np.where(Pxxx_welch >= half_max)[0]

            bw_start = ff_welch[above_half[0]]
            bw_end = ff_welch[above_half[-1]]
            bw = bw_end - bw_start
            rmssq = round(np.trapz(Pxxx_welch, ff_welch), 2)

            ad_v["accel_data"].update({'mean': mean, 'rms': rms, 'peak': peak})
            # ad_v["fs"] = fs

            psd_data['x'] = pd.Series(data=ff_welch)
            # psd_data['y']['fs'] = fs
            psd_data['y'][ad_k]["data"] = pd.Series(data=Pxxx_welch)
            psd_data['y'][ad_k]["psd_data"].update({'peak_welch': peak_value, 'rmssq': rmssq, 'bw': bw, 'dom_freq': df,
                                                    'npsg': npsg, 'res': avg})


class AC1(AccelerationData):
    def __init__(self, fdir, filename=None, **kwargs):
        super().__init__(fdir, filename=filename, **kwargs)

    def set_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.fdir, sep=';', decimal='.', names=['ind', 'time', 'x', 'y', 'z'], header=None)
        df.columns = df.iloc[0]
        df = df[1:]
        return df

    def set_datas(self) -> Datas:
        x_data = self.dataframe['t'].astype(float) / 1000000
        fs = 4096
        datas: Datas = {
            "datas": {
                DataType.accel: {'x': x_data,
                                 'y': {'accX': {'data': pd.Series(data=self.dataframe['AccX'].astype(float)),
                                                'kind': DataType.accel, 'fs': fs},
                                       'accY': {'data': pd.Series(data=self.dataframe['AccY'].astype(float)),
                                                'kind': DataType.accel, 'fs': fs},
                                       'accZ': {'data': pd.Series(data=self.dataframe['AccZ'].astype(float) - 1.0),
                                                'kind': DataType.accel, 'fs': fs}}},
                DataType.psd: {'x': x_data,
                               'y': {'accX': {'kind': DataType.psd, 'fs': fs},
                                     'accY': {'kind': DataType.psd, 'fs': fs},
                                     'accZ': {'kind': DataType.psd, 'fs': fs}}}}

            }

        self.__set_data_dict(datas)
        return cast(Datas, datas)


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

    def set_dataframe(self) -> pd.DataFrame | dict[str, pd.DataFrame] :
        df_dict: dict[str, pd.DataFrame] = {}
        for dir_k, dir_v in self.fdirs:
            df_dict[dir_k] = pd.read_csv(dir_v, header=None,
                                         names=['Hz', 'g^2/Hz', 'c3', 'c4', 'c5'] if dir_k == DataType.psd else
                                         ['time', 'accel', 'c3', 'c4', 'c5'],
                                         encoding='ISO-8859-1', sep='\t', decimal='.',
                                         skiprows=43 if dir_k == DataType.psd else 33)
        return df_dict

    def set_datas(self, nperseg=5):
        time =  self.dataframe[DataType.accel]['time']
        accel = self.dataframe[DataType.accel]['accel']
        datas: Datas = {
            "datas": {
                DataType.accel: {'x': time, 'y': {'accZ': accel, 'kind': DataType.accel}},
                DataType.psd: {'x': self.dataframe[DataType.psd]['Hz'],
                               'y': {'data': self.dataframe[DataType.psd]['g^2/Hz'], }}}

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
    
