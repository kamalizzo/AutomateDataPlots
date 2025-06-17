import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from abc import ABC, abstractmethod, ABCMeta


from src.plot.DataProcessing import ECData, Plot, DFType


class AccelerationData(ECData, ABC):
    def __new__(cls, fdir, filename=None, curves=None, data_source=None):
        if data_source == "MEMS" or data_source is None:
            return super(AccelerationData, cls).__new__(AC1)
        else:
            return super(AccelerationData, cls).__new__(AC2)
        
    def __init__(self, fdir, filename=None, **kwargs):
        super().__init__(fdir, filename, **kwargs)

    @classmethod
    def _set_datatype(cls):
        return 'vib'    

    

class AC1(AccelerationData):
    def __init__(self, fdir, filename=None, curves=None, **kwargs):
        super().__init__(fdir, filename, curves, **kwargs)

    def set_dataframe(self, fname, fdir=None, separator=",", decimal='.', ) -> pd.DataFrame:
        return pd.read_csv(fname, sep=separator, decimal=decimal, skiprows=1)
    
    def generate_list_curves(self):
        return super().generate_list_curves()
    
    def generate_full_dict(self):
        return super().generate_full_dict()
    
    def generate_chosen_dict(self):
        dict_chosen = {}

        return super().generate_chosen_dict()
    
    def generate_datafigure(self):
        return super().generate_datafigure()


class AC2(AccelerationData):
    def __init__(self, fdir, filename=None, curves=None, **kwargs):
        super().__init__(fdir, filename, curves, **kwargs)

    def set_dataframe(self, fname, fdir=None, separator='\t', decimal='.', ) -> pd.DataFrame:
        return pd.read_csv(fname, sep=separator, decimal=decimal, skiprows=42)
    
    def generate_list_curves(self):
        return super().generate_list_curves()
    
    def generate_full_dict(self):
        return super().generate_full_dict()
    
    def generate_chosen_dict(self):
        dict_chosen = {}

        return super().generate_chosen_dict()
    
    def generate_datafigure(self):
        return super().generate_datafigure()        
    
