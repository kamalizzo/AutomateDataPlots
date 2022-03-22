import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
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
    def __init__(self, fdir, filename=None, curves=None, test_stand=1,
                 **kwargs):
        self.wdir = fdir
        self.fdir = glob.glob(f"{fdir}/*" + "*.txt") \
            if fdir[-4:] != '.txt' else fdir
        self.dataframe = \
            pd.concat(
                [pd.read_csv(datas, encoding='ISO-8859-1', sep='\t',
                             decimal=',')
                 for datas in self.fdir], ignore_index=True)
        self.index_list = [_ + 1 for _ in range(curves)] if curves is not None \
            else self.generate_list_curves()
        self.fname = fdir.split('/')[-1][:-6] if not filename else filename
        self.data_type = self._set_datatype()
        self.generated_dict = {}
        self.generated_figures = {}
        self.test_stand = test_stand
        self.columns = kwargs.pop('column', [])
        self.dict_chosen_df, self.dict_full_df = self.generate_dict()
        self.generate_measurement_plots()

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
    def generate_list_curves(self):
        pass

    @abstractmethod
    def generate_dict(self):
        pass

    def generate_measurement_plots(self):
        fig = plt.figure(figsize=(13, 9), dpi=136)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        timeplot = self.dataframe['T relativ [min]']
        firstaxis = ['AI.T.Air.ST.Hum.1 [°C]', 'AI.T.H2.ST.Hum.2 [°C]',
                     'AI.T.Air.ST.UUT.out [°C]', 'I Summe [A]']
        secondaxis = ['AI.P.Air.SP.Ca.in [bar]', 'AI.P.H2.SP.An.in [bar]',
                      'AI.U.E.Co.Tb.1 [V]']
        [ax1.plot(timeplot, self.dataframe[data], label=data) for data in
         firstaxis]
        [ax2.plot(timeplot, self.dataframe[data], label=data) for
         data in secondaxis]

        line, label = ax1.get_legend_handles_labels()
        line2, label2 = ax2.get_legend_handles_labels()
        ax2.legend(Plot.flip(line + line2, 4), Plot.flip(label + label2, 4),
                   loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=4)
        plt.grid(True)
        ax1.set_xlim([0, 2000])
        ax1.set_xlabel('Time [min]')
        ax1.set_ylim([0, 100])
        ax1.set_ylabel('T [°C]; I [A]')
        ax2.set_ylim([0, 2.4])
        ax2.set_ylabel('Prel [bar]; U [V]')
        self.generated_figures.\
            update({f'{self.fname} Measurement Plots':
                    {'axis': 'multi', 'fig': fig, 'kw': 'Ges_Messverlauf'}})


class PolCurveData(ECData):
    def __init__(self, fdir, filename=None, curves=None, test_stand=1):
        super().__init__(fdir, filename, curves, test_stand)
        self.generate_datafigure()
        self.generate_polarization_curve()
        self.generate_hfr_curve()

    @classmethod
    def _set_datatype(cls):
        return 'pc'

    def generate_list_curves(self):
        new_df = self.dataframe
        num_polcurve_list = []
        for polar in new_df['Kommentar']:
            if 'Pol' in polar and 'OCV' not in polar:
                pol = int(polar[-1])
                if pol not in num_polcurve_list:
                    num_polcurve_list.append(pol)
        return num_polcurve_list

    def generate_dict(self):
        dict_pol = {}
        full_dict_pol = {}
        pol_list = self.index_list

        new_df = self.dataframe
        if self.test_stand == 1:
            for polar in pol_list:
                index_pol = []
                for num, marker in enumerate(new_df['SetMarker']):
                    if marker == polar:
                        index_pol.append(num)
                full_dict_pol[f'{polar}'] = new_df.iloc[index_pol]
                new_index_pol = []
                check_list = [[new_df.iloc[index_pol[0], 4],
                               new_df.iloc[index_pol[0], 3]]]

                for num, ind in enumerate(index_pol):
                    check_val = [new_df.iloc[ind, 4], new_df.iloc[ind, 3]]
                    if check_val not in check_list:
                        check_list.append(check_val)
                        new_index_pol.append(index_pol[num - 1])
                    else:
                        if num == len(index_pol) - 1:
                            new_index_pol.append(ind)

                dict_pol[f'{polar}'] = new_df.iloc[new_index_pol]
        return dict_pol, full_dict_pol

    def generate_polarization_curve(self, xrange=(0, 5), yrange=(0, 1)):
        fig = plt.figure(figsize=(13, 9), dpi=136)
        ax1 = fig.add_subplot(111)

        for key, val in self.dict_chosen_df.items():
            cur = val['I Summe [A]'] / 25
            vol = val['AI.U.E.Co.Tb.1 [V]']
            ax1.plot(cur, vol, label=f'Pol{key}')
            ax1.scatter(cur, vol)
        line, label = ax1.get_legend_handles_labels()
        ax1.legend(line, label, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=6)
        plt.grid(True)
        ax1.set_xlim(xrange)
        ax1.set_xlabel('Current [A.cm-2]')
        ax1.set_ylim(yrange)
        ax1.set_ylabel('Voltage [V]')
        self.generated_figures. \
            update({f'{self.fname} Polarisation Curves':
                    {'axis': 'single', 'fig': fig, 'kw': 'PolKurven'}})

    def generate_hfr_curve(self, xrange=(0, 5), yrange=(0, 100)):
        fig = plt.figure(figsize=(13, 9), dpi=136)
        ax1 = fig.add_subplot(111)

        for key, val in self.dict_chosen_df.items():
            axis = []
            cur = val['I Summe [A]'] / 25
            hfr = val['HFR [mOhm]']
            for h, c in zip(hfr, cur):
                if h > 0:
                    axis.append([c, h * 25])
            ax_df = pd.DataFrame(axis, columns=['CD', 'HFR'])
            ax1.plot(ax_df.values[:, 0], ax_df.values[:, 1], label=f'Pol{key}')
            ax1.scatter(ax_df.values[:, 0], ax_df.values[:, 1])
        line, label = ax1.get_legend_handles_labels()
        ax1.legend(line, label, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=6)
        plt.grid(True)
        ax1.set_xlim(xrange)
        ax1.set_xlabel('Current [A.cm-2]')
        ax1.set_ylim(yrange)
        ax1.set_ylabel('HFR [mOhm.cm²]')
        self.generated_figures. \
            update({f'{self.fname} High Frequency Resistance':
                    {'axis': 'single', 'fig': fig, 'kw': 'HFR'}})

    def generate_datafigure(self, xrange=(0, 5), yrange=(0, 1),
                            y2range=(0, 100)):
        fig = plt.figure(figsize=(13, 9), dpi=136)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        for key, val in self.dict_chosen_df.items():
            axis2 = []
            # if self.test_stand == 1:
            cur = val['I Summe [A]'] / 25
            vol = val['AI.U.E.Co.Tb.1 [V]']
            hfr = val['HFR [mOhm]']
            # else:
            #     data_column = self.columns
            #     cur = val[:, data_column[0]]
            #     vol = val[:, data_column[1]]
            #     hfr = vol / cur
            for h, c in zip(hfr, cur):
                if h > 0:
                    axis2.append([c, h * 25])
            ax2_d = pd.DataFrame(axis2, columns=['CD', 'HFR'])
            ax1.plot(cur, vol, label=f'Pol{key}')
            ax1.scatter(cur, vol)
            ax2.plot(ax2_d.values[:, 0], ax2_d.values[:, 1], label=f'HFR{key}',
                     linestyle='--')
            ax2.scatter(ax2_d.values[:, 0], ax2_d.values[:, 1])

        line, label = ax1.get_legend_handles_labels()
        line2, label2 = ax2.get_legend_handles_labels()
        ax2.legend(Plot.flip(line + line2, 6), Plot.flip(label + label2, 6),
                   loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=6)
        plt.grid(True)
        ax1.set_xlim(xrange)
        ax1.set_xlabel('Current [A.cm-2]')
        ax1.set_ylim(yrange)
        ax1.set_ylabel('Voltage [V]')
        ax2.set_ylim(y2range)
        ax2.set_ylabel('HFR [mOhm.cm²]')
        self.generated_figures. \
            update({f'{self.fname} Polarisation Curve and HFR':
                    {'axis': 'multi', 'fig': fig, 'kw': 'PolKurven'}})


class EISData(ECData):
    def __init__(self, fdir, filename=None, curves=None, test_stand=1):
        super().__init__(fdir, filename, curves, test_stand)

    @classmethod
    def _set_datatype(cls):
        return 'eis'

    def generate_list_curves(self):
        pass

    def generate_dict(self):
        pass


class EISSimData:
    pass
