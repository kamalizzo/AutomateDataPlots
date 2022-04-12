import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod, ABCMeta


from src.DataProcessing import ECData, Plot


class PolCurveData(ECData, ABC):
    def __new__(cls, fdir, filename=None, curves=None, test_stand=None):
        if test_stand == 'Abteilung4' or test_stand is None:
            return super(PolCurveData, cls,).__new__(PC1)
        else:
            return super(PolCurveData, cls).__new__(PC2)

    def __init__(self, fdir, filename=None, curves=None):
        super().__init__(fdir, filename, curves)

    @classmethod
    def _set_datatype(cls):
        return 'pc'

    @abstractmethod
    def generate_polarization_curve(self, xrange=(0, 5), yrange=(0, 1)):
        pass


class PC1(PolCurveData):
    def __init__(self, fdir, filename=None, curves=None):
        super().__init__(fdir, filename, curves)
        self.generate_datafigure()

    def set_dataframe(self, encoding='ISO-8859-1', decimal=','):
        return \
            pd.concat([pd.read_csv(datas, encoding=encoding, sep='\t',
                                   decimal=decimal)
                       for datas in self.fdir], ignore_index=True)

    def generate_list_curves(self, curve='Pol'):
        new_df = self.dataframe
        num_polcurve_list = []
        for polar in new_df['Kommentar']:
            if curve in polar and 'OCV' not in polar:
                pol = int(polar[-1])
                if pol not in num_polcurve_list:
                    num_polcurve_list.append(pol)
        return num_polcurve_list

    def generate_full_dict(self):
        # dict_pol = {}
        full_dict_pol = {}
        pol_list = self.index_list

        new_df = self.dataframe
        for polar in pol_list:
            index_pol = []
            for num, marker in enumerate(new_df['SetMarker']):
                if marker == polar:
                    index_pol.append(num)
            full_dict_pol[f'{polar}'] = new_df.iloc[index_pol]
            # index_pol_cpy = index_pol.copy()
            # new_index_pol = []  # To append chosen dict
            # for i_ind in index_pol:
            #     check_ind_i = new_df.iloc[i_ind, -14]
            #     if check_ind_i >= 0:
            #         continue
            #     else:
            #         index_pol_cpy.remove(i_ind)
            #
            # check_list = [[new_df.iloc[index_pol_cpy[0], 4],
            #                new_df.iloc[index_pol_cpy[0], 3]]]
            #
            # for num, ind in enumerate(index_pol_cpy):
            #     check_val = [new_df.iloc[ind, 4],
            #                  # new_df.iloc[ind, 3]
            #                  ]
            #     if check_val not in check_list:
            #         check_list.append(check_val)
            #         new_index_pol.append(index_pol_cpy[num - 1])
            #     else:
            #         # end of line
            #         if num == len(index_pol_cpy) - 1:
            #             new_index_pol.append(ind)
            # dict_pol[f'{polar}'] = new_df.iloc[new_index_pol]
        return full_dict_pol

    def generate_chosen_dict(self):
        dict_chosen = {}
        for key, value in self.dict_full_df.items():
            # ind_chosen = []
            # cur = value['I Mittel [A]']
            # vol = value['U Mittel [V]']
            # hfr = value['HFR [mOhm]']
            # for ind, (c, v, h) in enumerate(zip(cur, vol, hfr)):
            #     if h > 0:
            #         ind_chosen.append(ind)

            # new_df = value.iloc[ind_chosen]
            tme = value['T relativ [min]']
            cur = value['I Mittel [A]']
            vol = value['U Mittel [V]']
            hfr = value['HFR [mOhm]']
            t_bef = array_bef = total_array = hfr_bef = hfr_co = 0
            for n_ind, (t, c, v, h) in enumerate(zip(tme, cur, vol, hfr)):
                # new_index_pol = []  # To append chosen dict

                hfr_now = (hfr_bef + h)/2 if hfr_bef > 0 else h
                array_now = [[c, v, hfr_now]]
                if n_ind == 0:
                    hfr_bef = h
                    t_bef = t
                    array_bef = array_now
                    continue

                if t-t_bef > 1:
                    if isinstance(total_array, np.ndarray):
                        if hfr_bef < 0:
                            total_array = \
                                np.append(total_array,
                                          [[array_bef[0][0],array_bef[0][1],
                                           hfr_co]],
                                          axis=0)
                        else:
                            total_array = \
                                np.append(total_array, array_bef, axis=0)
                    else:
                        total_array = np.array(array_bef)

                    hfr_co = h

                if n_ind == len(value):
                    total_array = \
                        np.append(total_array, array_now, axis=0)

                hfr_bef = h
                array_bef = array_now

                t_bef = t

            dict_chosen[key] = \
                pd.DataFrame(total_array, columns=['CD', 'V', 'HFR'])

        return dict_chosen

    def generate_polarization_curve(self, xrange=(0, 5), yrange=(0, 1)):
        fig = plt.figure(figsize=(13, 9), dpi=136)
        ax1 = fig.add_subplot(111)

        for key, val in self.dict_chosen_df.items():
            cur = val['CD'] / 25
            vol = val['V']
            ax1.plot(cur, vol, label=f'Pol{key}')
            ax1.scatter(cur, vol)
        line, label = ax1.get_legend_handles_labels()
        ax1.legend(line, label, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=6)
        ax1.minorticks_on()
        plt.grid(True)
        ax1.grid(True, which='minor', color='lightgrey', linestyle='--')
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
            cur = val['CD'] / 25
            hfr = val['HFR']
            for h, c in zip(hfr, cur):
                if h > 0:
                    axis.append([c, h * 25])
            ax_df = pd.DataFrame(axis, columns=['CD', 'HFR'])
            # ax1.plot(cur, hfr, label=f'Pol{key}')
            # ax1.scatter(cur, hfr)
            ax1.plot(ax_df.values[:, 0], ax_df.values[:, 1], label=f'Pol{key}')
            ax1.scatter(ax_df.values[:, 0], ax_df.values[:, 1])
        # line, label = ax1.get_legend_handles_labels()
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=6)
        ax1.minorticks_on()
        plt.grid(True)
        ax1.grid(True, which='minor', color='lightgrey', linestyle='--')
        ax1.set_xlim(xrange)
        ax1.set_xlabel('Current [A.cm-2]')
        ax1.set_ylim(yrange)
        ax1.set_ylabel('HFR [mOhm.cm²]')
        self.generated_figures. \
            update({f'{self.fname} High Frequency Resistance':
                        {'axis': 'single', 'fig': fig, 'kw': 'HFR'}})

    def generate_pc_hfr(self, xrange=(0, 5), yrange=(0, 1), y2range=(0, 100)):
        fig = plt.figure(figsize=(13, 9), dpi=136)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        for key, val in self.dict_chosen_df.items():
            axis2 = []
            cur = val['CD'] / 25
            vol = val['V']
            hfr = val['HFR']
            for h, c in zip(hfr, cur):
                if h > 0:
                    axis2.append([c, h * 25])
            ax_df = pd.DataFrame(axis2, columns=['CD', 'HFR'])
            hfr = val['HFR']
            ax1.plot(cur, vol, label=f'Pol{key}')
            ax1.scatter(cur, vol)
            # ax2.plot(cur, hfr, label=f'HFR{key}',
            #          linestyle='--')
            # ax2.scatter(cur, hfr)
            ax2.plot(ax_df.values[:, 0], ax_df.values[:, 1], label=f'HFR{key}',
                     linestyle='--')
            ax2.scatter(ax_df.values[:, 0], ax_df.values[:, 1])

        line, label = ax1.get_legend_handles_labels()
        line2, label2 = ax2.get_legend_handles_labels()
        ax2.legend(Plot.flip(line + line2, 6), Plot.flip(label + label2, 6),
                   loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=6)
        ax1.grid(True)
        ax1.set_xlim(xrange)
        ax1.set_xlabel('Current [A.cm-2]')
        ax1.set_ylim(yrange)
        ax1.set_ylabel('Voltage [V]')
        ax2.set_ylim(y2range)
        ax2.set_ylabel('HFR [mOhm.cm²]')
        self.generated_figures. \
            update({f'{self.fname} Polarisation Curve and HFR':
                    {'axis': 'multi', 'fig': fig, 'kw': 'PolKurven'}})

    def generate_measurement_plots(self):
        fig = plt.figure(figsize=(13, 9), dpi=136)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        # last_ind = self.dataframe.index[-1]
        timeplot = self.dataframe['T relativ [min]']
        firstaxis = ['AI.T.Air.ST.Hum.1 [°C]', 'AI.T.H2.ST.Hum.2 [°C]',
                     'AI.T.Air.ST.UUT.out [°C]', 'I Summe [A]']
        secondaxis = ['AI.P.Air.SP.Ca.in [bar]', 'AI.P.H2.SP.An.in [bar]',
                      'AI.U.E.Co.Tb.1 [V]']
        [ax1.plot(timeplot, self.dataframe[data], label=data,
                  color=col) for (data, col) in
         zip(firstaxis, ['grey', 'yellow', 'lightblue', 'darkblue'])]
        [ax2.plot(timeplot, self.dataframe[data], label=data,
                  # linestyle='dashed',
                  color=col) for
         (data, col) in zip(secondaxis, ['tab:blue', 'orange', 'tab:green'])]

        line, label = ax1.get_legend_handles_labels()
        line2, label2 = ax2.get_legend_handles_labels()
        ax2.legend(Plot.flip(line + line2, 4), Plot.flip(label + label2, 4),
                   loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=4)
        ax1.grid(True)
        ax1.set_xlim([0, 2700])
        ax1.set_xlabel('Time [min]')
        ax1.set_ylim([0, 100])
        ax1.set_ylabel('T [°C]; I [A]')
        ax2.set_ylim([0, 2.4])
        ax2.set_ylabel('Prel [bar]; U [V]')
        self.generated_figures.\
            update({f'{self.fname} Measurement Plots':
                    {'axis': 'multi', 'fig': fig, 'kw': 'Ges_Messverlauf'}})

    def generate_datafigure(self):
        self.generate_measurement_plots()
        self.generate_pc_hfr()
        self.generate_polarization_curve()
        self.generate_hfr_curve()


class PC2(PolCurveData):
    def __init__(self, fdir, measured_points, filename=None, curves=6):
        super().__init__(fdir, filename, curves)
        self.data_columns = \
            ['Time', 'H2 [ml/min]', 'H2O An [g/h]', 'Druckluft [ml/min]',
             'H2O Ca [g/h]', 'N2 An [ml/min]', 'N2 Ka [ml/min]', 'Voltage [V]',
             'Current [A]', 'Power [W]', 'Temp1', 'Temp2', 'Temp3', 'Temp4',
             'Temp5', 'Temp6', 'Temp7', 'Temp8', 'Press1', 'Press2', 'Press3',
             'Press4', 'StoeH2', 'StoeL', 'U1']
        self.measured_points = measured_points \
            if isinstance(measured_points, list) else [measured_points]
        # [0, 30, 28, 26, 24, 20 , 15]

    def set_dataframe(self, fdir=None):
        return pd.read_csv(self.fdir, encoding='ISO-8859-1', sep='\t',
                           names=self.data_columns)

    def generate_list_curves(self):
        pass

    def generate_full_dict(self):
        pass

    def generate_chosen_dict(self):
        pass

    def generate_polarization_curve(self, xrange=(0, 5), yrange=(0, 1)):
        pass

    def generate_datafigure(self):
        pass
    