import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod, ABCMeta
from typing import Mapping, TypedDict, Union, Literal, NotRequired, Dict
import itertools
from dataclasses import dataclass
from enum import Enum
from deprecated import deprecated


class Plot:

    @staticmethod
    def flip(items, ncol):
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])

    @staticmethod
    def show(func):
        def inner():
            func()
            plt.show()

        return inner


class DFType(Enum):
    txt = ".txt"
    csv = ".csv"


class DataType(Enum):
    accel = "acceleration"
    psd = "power_spectral_density"
    polcurve = "polarisation_curve"
    eis = "electrochemical_impedance_spectroscopy"


class YDataSets(TypedDict):
    data: pd.Series

class YDataSetsAccel(YDataSets):
    kind: Literal[DataType.accel]
    accel_data: NotRequired[Dict[str, float]]
    fs: float

class YDataSetsPSD(YDataSets):
    kind: Literal[DataType.psd]
    psd_data: NotRequired[Dict[str, float]]
    fs: float

class Datasets(TypedDict):
    x: pd.Series
    y: Dict[str, YDataSetsAccel | YDataSetsPSD] | YDataSetsAccel | YDataSetsPSD


class Datas(TypedDict):
    datas: Dict[DataType, Datasets]



class ECData(metaclass=ABCMeta):
    def __init__(self, fdir, fdirs: dict[str, str] = None, filename=None, curves=None, **kwargs):
        self.wdir: str = fdir
        self.fdir: str = None if not fdirs else fdir
        self.fdirs: list[str] | dict[str, str] = self.access_fdir(fdir) if fdirs is None else fdirs
        self.file_list: list = self.generate_file_list()
        self.dataframe = self.set_dataframe()
        self.index_list: list = [_ + 1 for _ in range(curves)] if curves is not None \
            else self.generate_list_curves()
        self.fname = fdir.split('/')[-1][:-6] if not filename else filename
        self._data_type = self._set_datatype()

        self.generated_dict = {}
        self.generated_figures = {}
        self.data_filetype: str = kwargs.pop('dftype', DFType.txt.name)
        self.columns = kwargs.pop('column', [])
        self.dict_full_df = self.generate_full_dict()
        self.dict_chosen_df = self.generate_chosen_dict()

        self.datas: Datas = self.set_datas()

    @staticmethod
    def access_fdir(fdir):
        return glob.glob(f"{fdir}/01*" + "*.txt") if fdir[-4:] != '.txt' \
            else [fdir]

    def generate_file_list(self):
        f_list = []
        for fdir_str in self.fdirs:
            new_str = fdir_str.split('/')[-1].split('\\')[-1]
            f_list.append(new_str)
        return f_list

    @deprecated(reason="Use generate_file_list")
    def gen_file_list(self):
        file_list = []
        for file in self.fdirs:
            if file[:2] not in file_list:
                file_list.append(file[:2])
        return file_list[0]

    @abstractmethod
    def _set_datatype(self):
        pass

    @abstractmethod
    def set_dataframe(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        pass

    @abstractmethod
    def set_datas(self):
        pass

    @abstractmethod
    def set_datasets(self):
        pass

    @abstractmethod
    def generate_list_curves(self, **kwargs):
        pass

    @abstractmethod
    def generate_full_dict(self, **kwargs):
        pass

    @abstractmethod
    def generate_chosen_dict(self, **kwargs):
        pass

    @abstractmethod
    def generate_datafigure(self, **kwargs):
        pass
