import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.DataProcessing import ECData, Plot
from src import DataStr as ds, GenerateData as gd


class EISData(ECData):
    def __init__(self, fdir, filename=None, curves=None, shift=0):
        super().__init__(fdir, filename, curves)
        # self.phase_shift_corrector(shift)
        self.generate_datafigure()

    @classmethod
    def _set_datatype(cls):
        return 'eis'

    def set_dataframe(self, fdir=None):
        files = self.access_fdir(fdir) if fdir else self.fdir
        fnames = [os.path.basename(x)
                  for x in glob.glob(f'{self.wdir}/*' + "*.txt")]
        eis_dict = {}
        for f, fn in zip(files, fnames):
            fname = fn.split(',')[0].split('.')  # e.g EIS_1.1_25A
            df = pd.read_csv(f, encoding='ISO-8859-1', sep='\t', decimal=',',
                             skiprows=21, low_memory=False)
            if fname[0][3:] in eis_dict:
                eis_dict[fname[0][3:]].update({fname[1]: df})
            else:
                eis_dict[fname[0][3:]] = {fname[1]: df}

        return eis_dict

    def generate_list_curves(self):
        pass

    def generate_full_dict(self):
        return self.dataframe

    def generate_chosen_dict(self):
        return self.dataframe

    def gamry_to_zplot(self):
        gd.GenerateData.open_plots_folder(self.wdir, 'to_zplot')
        for (fd, fl) in zip(self.fdir, self.file_list):
            df_gamry = pd.read_csv(fd, encoding='ISO-8859-1', sep='\t',
                                   decimal=',', skiprows=21, low_memory=False)
            with open(os.getcwd()+'/'+fl[:-4]+'_to_zplot.txt', 'w') as f:
                for line in ds.zplot_str:
                    f.write(line)
                for (freq, real, imag) in zip(df_gamry['freq [Hz]'],
                                              df_gamry["real [ohm]"],
                                              df_gamry["imag [ohm]"]):
                    f.write(f'{round(freq, 2)}\t0.00E+00\t0'
                            f'.00E+00\t0\t{round(real * 25, 6)}\t'
                            f'{"{:.2e}".format(imag * 25)}\t0'
                            f'.00E+00\t0\t0\n')

    @staticmethod
    def to_bode(df_dict, shift=0, signal='1'):
        eis_dict = {}
        curve = str(signal) if not isinstance(signal, str) else signal
        if isinstance(df_dict, pd.DataFrame):
            real = df_dict['real [ohm]']
            imag = df_dict['imag [ohm]']
            magnitude = np.sqrt(real + imag)
            pshift = np.arctan(imag / real)
            for i, val in enumerate(pshift):
                pshift[i] = abs(val+shift)
            eis_dict.update(
                {f'{curve}':
                 pd.DataFrame({'freq meas [Hz]': df_dict['freq meas [Hz]'],
                               'real [ohm]': real, 'imag [ohm]': imag,
                               'impedance [ohm]': magnitude,
                               'phase [deg]': pshift})})
        elif isinstance(df_dict, dict):
            for key, val in df_dict.items():
                EISData.to_bode(val, shift, key)
        return eis_dict

    def generate_bodes(self,  xrange=(0.1, 50100),
                       yrange=(0.0005, 0.03), y2range=(0, 90)):
        eis_df = self.dataframe  # if df is None else df - df: dict = None,
        for key, val in eis_df.items():
            fig = plt.figure(figsize=(13, 9), dpi=136)
            ax1 = fig.add_subplot(111)
            ax1.semilogx()
            ax2 = ax1.twinx()
            # eis_df = self.to_bode(val, key)
            num_col = 0
            for k, signal in val.items():
                num_col += 1
                freq = signal['freq meas [Hz]']
                imp = signal['impedance [ohm]']
                pshift = signal['phase [deg]']
                for i, abs_v in enumerate(pshift):
                    new_v = abs_v
                    pshift[i] = abs(new_v)
                ax1.plot(freq, imp, label=f'{key}-{k[0]}A |Z|')
                ax1.scatter(freq, imp, marker='.')
                ax2.plot(freq, pshift, label=f'{key}-{k[0]}A |°|',
                         linestyle='--')
                ax2.scatter(freq, pshift,  marker='+')

            line, label = ax1.get_legend_handles_labels()
            line2, label2 = ax2.get_legend_handles_labels()
            ax1.legend(Plot.flip(line+line2, num_col),
                       Plot.flip(label+label2, num_col), loc='upper center',
                       bbox_to_anchor=(0.5, -0.06), fancybox=True, shadow=True,
                       ncol=num_col)

            ax1.grid(True)
            ax2.grid(linestyle='--')
            ax1.set_xlim(xrange)
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylim(yrange)
            ax1.set_ylabel('|Impedance| [|Z|]')
            ax2.set_ylim(y2range)
            ax2.set_ylabel('|Phase Shift| [°]')
            self.generated_figures. \
                update({f'{self.fname} {key} Bode Plots':
                        {'axis': 'multi', 'fig': fig, 'kw': 'EISKurven'}})
            plt.close(fig)

    def generate_bodes_current(self, xrange=(0.1, 50100),
                               yrange=(0.0005, 0.03), y2range=(0, 90)):
        eis_df = self.dataframe  # if df is None else df - df: dict = None,
        for cur in [1, 2, 3]:
            num_col = 0
            fig = plt.figure(figsize=(13, 9), dpi=136)
            ax1 = fig.add_subplot(121)
            ax1.semilogx()
            ax2 = fig.add_subplot(122)
            ax2.semilogx()
            # ax2 = ax1.twinx()
            for key, val in eis_df.items():
                for k, signal in val.items():
                    if k[0] == str(cur):
                        num_col += 1
                        freq = signal['freq meas [Hz]']
                        imp = signal['impedance [ohm]']
                        pshift = signal['phase [deg]']
                        for i, abs_v in enumerate(pshift):
                            pshift[i] = abs(abs_v)
                        ax1.plot(freq, imp, label=f'{key}-{k[0]}A')
                        ax1.scatter(freq, imp, marker='.')
                        ax2.plot(freq, pshift)
                                 # linestyle='--')
                        ax2.scatter(freq, pshift, marker='+')
            line, label = ax1.get_legend_handles_labels()
            # line2, label2 = ax2.get_legend_handles_labels()
            plt.legend(line, label, loc='upper center',
                       bbox_to_anchor=(-0.1, -0.06), ncol=num_col)

            ax1.grid(True)
            ax2.grid(True)
            ax1.set_xlim(xrange)
            ax1.set_xlabel('Frequency [Hz]')
            ax2.set_xlim(xrange)
            ax2.set_xlabel('Frequency [Hz]')
            ax1.set_ylim(yrange)
            ax1.set_ylabel('|Impedance| [|Z|]')
            ax2.set_ylim(y2range)
            ax2.set_ylabel('|Phase Shift| [°]')
            self.generated_figures. \
                update({f'{self.fname} {cur}A Bode Plots':
                        {'axis': 'multi', 'fig': fig, 'kw': 'EISKurven'}})
            plt.close(fig)

    def to_nyquist(self, df_dict=None, shift=0, signal='1'):
        eis_dict = {}
        curve = str(signal) if not isinstance(signal, str) else signal
        if isinstance(df_dict, pd.DataFrame):
            pshift = df_dict['phase [deg]'] + shift
            impedanz = df_dict['impedance [ohm]']
            # for i, val in enumerate(pshift):
            #     pshift[i] = abs(val)
            eis_dict.update(
                {f'{curve}':
                 pd.DataFrame({'freq meas [Hz]': df_dict['freq meas [Hz]'],
                               'real [ohm]': impedanz * np.cos(pshift),
                               'imag [ohm]': impedanz * np.sin(pshift),
                               'impedance [ohm]': impedanz,
                               'phase [deg]': pshift})})
            return eis_dict
        elif isinstance(df_dict, dict):
            for key, val in df_dict.items():
                EISData.to_nyquist(val, shift, key)

    def generate_nyquists(self, xrange=(0.0005, 0.03),
                          yrange=(0.0005, -0.012)):
        eis_df = self.dataframe # if df is None else df - df: dict = None,
        for key, val in eis_df.items():
            fig = plt.figure(figsize=(13, 9), dpi=136)
            ax1 = fig.add_subplot(111)
            ax1.invert_yaxis()
            num_col = 0
            for k, signal in val.items():
                num_col += 1
                real = signal['real [ohm]']
                img = signal['imag [ohm]']
                ax1.plot(real, img, label=f'{key}-{k[0]}A/cm²')
                ax1.scatter(real, img, marker='.')

            line, label = ax1.get_legend_handles_labels()
            ax1.legend(line, label, loc='upper center',
                       bbox_to_anchor=(0.5, -0.06), ncol=num_col)

            ax1.grid(True)
            ax1.set_xlim(xrange)
            ax1.set_xlabel('Real Impedance Z-Re')
            ax1.set_ylim(yrange)
            ax1.set_ylabel('Imaginary Impedance Z-Im')
            self.generated_figures. \
                update({f'{self.fname} {key} Nyquist Plots':
                        {'axis': 'multi', 'fig': fig, 'kw': 'EISKurven'}})
            plt.close(fig)

    def generate_nyquists_current(self, xrange=(0.0005, 0.03),
                                  yrange=(0.0005, -0.012)):
        eis_df = self.dataframe # if df is None else df
        for cur in [1, 2, 3]:
            num_col = 0
            fig = plt.figure(figsize=(13, 9), dpi=136)
            ax1 = fig.add_subplot(111)
            ax1.invert_yaxis()
            for key, val in eis_df.items():
                for k, signal in val.items():
                    if k[0] == str(cur):
                        num_col += 1
                        real = signal['real [ohm]']
                        img = signal['imag [ohm]']
                        ax1.plot(real, img, label=f'{key}-{k[0]}A')
                        ax1.scatter(real, img, marker='.')
            line, label = ax1.get_legend_handles_labels()
            ax1.legend(line, label, loc='upper center',
                       bbox_to_anchor=(0.5, -0.06), ncol=num_col)

            ax1.grid(True)
            ax1.set_xlim(xrange)
            ax1.set_xlabel('Real Impedance Z-Re')
            ax1.set_ylim(yrange)
            ax1.set_ylabel('Imaginary Impedance Z-Im')
            self.generated_figures. \
                update({f'{self.fname} {cur}A Nyquist Plots':
                        {'axis': 'multi', 'fig': fig, 'kw': 'EISKurven'}})
            plt.close(fig)

    def generate_datafigure(self):
        self.generate_bodes()
        self.generate_bodes_current()
        self.generate_nyquists()
        self.generate_nyquists_current()

    def phase_shift_corrector(self, shift=0):
        new_dict = {}
        for key, val in self.dataframe.items():
            for k, v in val.items():
                pshift = v['phase [deg]'].to_numpy() + shift
                pshift_rad = np.radians(pshift)
                impedanz = v['impedance [ohm]'].to_numpy()

                real = impedanz * np.cos(pshift_rad)
                imag = impedanz * np.sin(pshift_rad)
                # for i, val in enumerate(pshift):
                #     pshift[i] = abs(val)
                if key in new_dict:
                    new_dict[key].update(
                        {k: pd.DataFrame(
                            {'freq meas [Hz]': v['freq meas [Hz]'].tolist(),
                             'real [ohm]': real.tolist(),
                             'imag [ohm]': imag.tolist(),
                             'impedance [ohm]': impedanz.tolist(),
                             'phase [deg]': pshift.tolist()})})
                else:
                    new_dict[key] = {k: pd.DataFrame(
                            {'freq meas [Hz]': v['freq meas [Hz]'].tolist(),
                             'real [ohm]': real.tolist(),
                             'imag [ohm]': imag.tolist(),
                             'impedance [ohm]': impedanz.tolist(),
                             'phase [deg]': pshift.tolist()})}
        self.dataframe = new_dict

    def generate_nyqs(self, xrange=(0.0005, 0.014), yrange=(0.0005, -0.004)):
        self.generate_nyquists(xrange, yrange)
        self.generate_nyquists_current(xrange, yrange)

    def generate_bods(self, xrange=(0.1, 50100), yrange=(0.0005, 0.0014),
                      y2range=(0, 90)):
        self.generate_bodes(xrange, yrange, y2range)
        self.generate_bodes_current(xrange, yrange)


class EISSimData:
    pass
