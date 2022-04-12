import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod, ABCMeta
import itertools


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


class ECData(ABC):
    def __init__(self, fdir, filename=None, curves=None, **kwargs):
        self.wdir = fdir
        self.fdir = self.access_fdir(fdir)
        self.dataframe = self.set_dataframe()
        self.index_list = [_ + 1 for _ in range(curves)] if curves is not None \
            else self.generate_list_curves()
        self.fname = fdir.split('/')[-1][:-6] if not filename else filename
        self._data_type = self._set_datatype()
        self.generated_dict = {}
        self.generated_figures = {}
        self.columns = kwargs.pop('column', [])
        self.dict_full_df = self.generate_full_dict()
        self.dict_chosen_df = self.generate_chosen_dict()

    @staticmethod
    def access_fdir(fdir):
        return glob.glob(f"{fdir}/*" + "*.txt") if fdir[-4:] != '.txt' \
            else [fdir]

    def generate_file_list(self):
        file_list = []
        for file in self.fdir:
            if file[:2] not in file_list:
                file_list.append(file[:2])
        return file_list[0]

    @abstractmethod
    def _set_datatype(self):
        pass

    @abstractmethod
    def set_dataframe(self, fdir=None):
        pass

    @abstractmethod
    def generate_list_curves(self):
        pass

    @abstractmethod
    def generate_full_dict(self):
        pass

    @abstractmethod
    def generate_chosen_dict(self):
        pass

    @abstractmethod
    def generate_datafigure(self):
        pass






