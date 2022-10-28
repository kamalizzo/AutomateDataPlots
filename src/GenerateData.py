import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from xlsxwriter.workbook import Workbook


class GenerateData:
    def __init__(self, obj):
        self.data_obj = obj
        self.folder_path = getattr(self.data_obj, 'wdir')
        self.fname: str = getattr(self.data_obj, 'fname')
        self.data_type = getattr(self.data_obj, '_data_type')
        self.dict_df = getattr(self.data_obj, 'dict_full_df')
        self.chosen_df = getattr(self.data_obj, 'dict_chosen_df')
        self.dataframe = getattr(self.data_obj, 'dataframe')
        self.generated_figures = getattr(self.data_obj, 'generated_figures')
        self.generated_path = self.generate_folder()
        self.generated_subfolder = self.generate_subfolder()
        self.generated_file_path = {}
        self.curve_type = self.__set_curve_type()
        # print(type(self.dict_df))
        # [print(type(d)) for k, d in self.dict_df.items()]

    def __set_curve_type(self):
        if self.data_type == 'pc':
            ctype = 'Pol'
        elif self.data_type == 'eis':
            ctype = 'EIS'
        else:
            raise NotImplementedError('DataType has not been implemented for '
                                      'GenerateData class')
        return ctype

    def generate_folder(self):
        os.chdir(self.folder_path)
        folder_n = 'generated'
        if not os.path.exists('generated'):
            os.makedirs('generated')
        return os.path.join(os.getcwd(), 'generated')

    def generate_subfolder(self):
        os.chdir(self.generated_path)
        if not os.path.exists(f'generated_{self.data_type}'):
            os.makedirs(f'generated_{self.data_type}')
        return os.path.join(os.getcwd(), f'generated_{self.data_type}')

    @staticmethod
    def open_plots_folder(path, folder):
        os.chdir(path)
        if not os.path.exists(f'{folder}'):
            os.makedirs(f'{folder}')
        return os.chdir(os.path.join(os.getcwd(), f'{folder}'))

    def generate_files(self, path=None, filetype=None):
        if filetype is None:
            filetype = ['pdf']
        if path is None:
            path = self.generated_subfolder
        for ftype in filetype:
            if ftype == 'xls':
                continue
            os.chdir(self.generated_subfolder)
            for title, file in self.generated_figures.items():
                self.open_plots_folder(path, ftype)
                fig = file['fig']
                fig.suptitle(f'{title}')
                fig.savefig(f'{title}.{ftype}')
                self.generated_file_path.\
                    update({f'{ftype}-{title}':
                            os.path.join(os.getcwd(), f'{title}.{ftype}')})

    # def generate_xls(self, filetype='xls'):
    #     self.open_plots_folder(self.generated_subfolder, filetype)
    #     self.generate_files(filetype=['png'])
    #     for title, file in self.generated_figures.items():
    #         if file['xls'] is True:
    #             workbook = Workbook(f'{title}.xlsx')
    #             worksheet = workbook.add_chartsheet(f'{kw}')
    #             worksheet.insert_image('A1', 'Hello world')
    #             workbook.close()

    def generate_raw_data(self, df_dict, path=None):
        if path is None:
            path = self.generated_path
        self.open_plots_folder(path, 'generated_rawdata')
        writer = pd.ExcelWriter(f'{self.fname}_rawdata.xlsx',
                                engine='xlsxwriter')
        for key, val in self.dict_df.items():
            val.to_excel(writer, sheet_name=f'generated_{key}')
        writer.save()

    def generate_all_raw_data(self, path=None):
        if path is None:
            path = self.generated_path
        self.open_plots_folder(path, 'generated_rawdata')
        writer = pd.ExcelWriter(f'{self.fname}_rawdata.xlsx',
                                engine='xlsxwriter')
        self.dataframe.to_excel(writer, sheet_name='Gesamt')
        for key, val in self.dict_df.items():
            val.to_excel(writer, sheet_name=f'{self.curve_type}{key}')
        writer.save()

    def generate_compiled_pdf(self, fname=None, path=None,
                              filetype='pdf'):
        if fname is None:
            fname = self.fname
        if path is None:
            path = self.generated_subfolder
        self.open_plots_folder(path, filetype)
        pdf = \
            matplotlib.backends.backend_pdf.PdfPages(f'compiled_{fname}'
                                                     f'.pdf')
        for title, file in self.generated_figures.items():
            fig = file['fig']
            fig_type = file['kw']
            fig.suptitle(f'{title} - {fig_type}')
            pdf.savefig(fig)
        pdf.close()

    def generate_all(self):
        self.open_plots_folder(self.generated_path, 'all')
        path = os.path.join(os.getcwd())
        # self.generate_compiled_pdf(path)
        self.generate_files(path, filetype=['pdf', 'png'])
        # self.generate_all_raw_data(path)

    def generate_files_only(self):
        self.open_plots_folder(self.generated_path, 'generated_pc')
        path = os.path.join(os.getcwd())
        self.generate_files(path, filetype=['pdf'])

