import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from pandas.plotting import table


class GasFlowCalc:
    constant_dict = \
        {'norm_tmp': 25, 'pressure': 1013.25, 'o2_mass_percentage': 23.135,
         'o2_vol_percentage': 20.942, 'faraday_c': 96485.3399,
         'avogadro_c': 6.02214e+23, 'mol_vol': 22.414, 'charge_o2': 4,
         'charge_h2': 2}
    const_param = \
        {'active_site': 25, 'tot_current': 100, 'num_cell': 1,
         'stoich_k': 2, 'stoich_a': 2, 'rhum_k': 100, 'rhum_a': 75,
         'overpressure': 0, 'cell_temp': 80, 'boiler_temp': 80}

    def __init__(self, stoich=None):
        self.c_dict = self.constant_dict
        self.c_param = self.const_param
        if stoich is not None:
            self.c_param.update({'stoich_a': stoich[0], 'stoich_k': stoich[1]})
        self.stack_verbrauch = self.calc_stack_verbrauch()
        self.o2_verbrauch = self.calc_gas_verbrauch('stoich_k')
        self.h2_verbrauch = self.calc_gas_verbrauch('stoich_a')
        self.o2_mol_s = self.calc_gas_mol_s('o2')
        self.h2_mol_s = self.calc_gas_mol_s('h2')
        self.o2_norml_s = self.calc_gas_norml_s('o2')
        self.h2_norml_s = self.calc_gas_norml_s('h2')
        self.air_flow = self.calc_air_flow()
        self.h2_flow = self.calc_h2_flow()

    def calc_stack_verbrauch(self, current=None):
        if current is None:
            stack_v = self.c_param['tot_current'] * self.c_param['num_cell']
        else:
            stack_v = current * self.c_param['num_cell']
        return stack_v

    def calc_gas_verbrauch(self, stoich):
        if isinstance(stoich, str):
            return self.stack_verbrauch * self.c_param[stoich]
        else:
            return self.stack_verbrauch * stoich

    def calc_gas_mol_s(self, gas):
        if gas == 'o2':
            gas_verbrauch = self.o2_verbrauch
        else:
            gas_verbrauch = self.h2_verbrauch
        return gas_verbrauch / self.c_dict['faraday_c'] /\
               self.c_dict[f'charge_{gas}']

    def calc_gas_norml_s(self, gas):
        if gas == 'o2':
            gas_flow = self.o2_mol_s
        else:
            gas_flow = self.h2_mol_s
        return gas_flow * self.c_dict['mol_vol']

    @staticmethod
    def calc_gas_nml_min(flow):
        return flow * 1000

    def calc_air_flow(self):
        flow_nl_min = \
            self.o2_norml_s * 60 / (self.c_dict['o2_vol_percentage'] / 100)
        return flow_nl_min

    def calc_h2_flow(self):
        return self.h2_norml_s * 60

    def calc_flows_air(self, stoich):
        self.o2_verbrauch = self.calc_gas_verbrauch(stoich)
        self.o2_mol_s = self.calc_gas_mol_s('o2')
        self.o2_norml_s = self.calc_gas_norml_s('o2')
        self.air_flow = self.calc_air_flow()
        return self.calc_gas_nml_min(self.air_flow)

    def calc_flows_h2(self, stoich):
        self.h2_verbrauch = self.calc_gas_verbrauch(stoich)
        self.h2_mol_s = self.calc_gas_mol_s('h2')
        self.h2_norml_s = self.calc_gas_norml_s('h2')
        self.h2_flow = self.calc_h2_flow()
        return self.calc_gas_nml_min(self.h2_flow)


class HumidityCalc(GasFlowCalc):
    water_dict = \
        {'charge_h2o': 2, 'molmass_h2o': 18.0153}
    wagner_constant = \
        {'c1': -5800.2206, 'c2': 1.3914993, 'c3': -0.048640239,
         'c4': 4.17648e-05, 'c5': -1.44521e-08, 'c6': 6.545973}

    def __init__(self, stoich=None, flow=None, r_hum=None):
        super().__init__(stoich)
        self.c_dict.update(self.water_dict)
        self.w_const = self.wagner_constant
        self.tmp = self.c_param['boiler_temp'] + 273.15
        self.overpressure = self.c_param['overpressure']
        self.rel_hum = self.set_relative_humidities(r_hum)
        self.o2_mol_min = self.calc_gas_mol_min('o2')
        self.n2_mol_min = self.calc_gas_mol_min('n2')
        self.sum_mol_min = self.o2_mol_min + self.n2_mol_min
        self.sat_pressure = self.calc_saturation_pressure()
        self.sat_pressure_mbar = self.sat_pressure / 100
        self.partial_pressures_h2o = self.sat_pressure_mbar * self.rel_hum / 100
        self.partial_pressures_o2 = self.calc_partial_pressure_gas('o2')
        self.partial_pressures_n2 = self.calc_partial_pressure_gas('n2')
        self.total_partials = self.calc_tot_partial_pressures()
        self.total_mol_mins = \
            self.n2_mol_min / self.partial_pressures_n2 * self.total_partials
        self.h2o_mol_mins = \
            (self.partial_pressures_h2o / self.total_partials) * \
            self.total_mol_mins
        self.h2o_flows = self.h2o_mol_mins * self.c_dict['molmass_h2o'] * 60

    # array values [anode, kathode]
    def set_relative_humidities(self, rhums=None):
        if rhums is None:
            rhum_array = \
                np.array([self.c_param['rhum_a'], self.c_param['rhum_k']])
        else:
            rhum_array = np.array([rhums[0], rhums[1]])
        return rhum_array

    def gas_anteil(self, gas):
        gas_anteil = self.c_dict['o2_vol_percentage'] if gas == 'o2' \
            else 100 - self.c_dict['o2_vol_percentage']
        return gas_anteil

    def calc_gas_mol_min(self, gas):
        return \
            self.air_flow * self.gas_anteil(gas) / 100 / self.c_dict['mol_vol']

    def calc_saturation_pressure(self):
        return np.exp(self.w_const['c1'] / self.tmp + self.w_const['c2'] +
                      self.w_const['c3'] * self.tmp +
                      self.w_const['c4'] * self.tmp ** 2 +
                      self.w_const['c5'] * self.tmp ** 3 +
                      self.w_const['c6'] * np.log(self.tmp))

    def calc_partial_pressure_gas(self, gas):
        return \
            (self.c_dict['pressure'] + self.overpressure -
             self.partial_pressures_h2o) * self.gas_anteil(gas) / 100

    def calc_tot_partial_pressures(self):
        return self.partial_pressures_n2 + self.partial_pressures_o2 + \
               self.partial_pressures_h2o

    def calc_flows_h2o(self, rhums, flow=None):
        if flow is not None:
            self.air_flow = flow
        self.o2_mol_min = self.calc_gas_mol_min('o2')
        self.n2_mol_min = self.calc_gas_mol_min('n2')
        self.rel_hum = self.set_relative_humidities(rhums)
        self.partial_pressures_h2o = self.sat_pressure_mbar * self.rel_hum / 100
        self.partial_pressures_o2 = self.calc_partial_pressure_gas('o2')
        self.partial_pressures_n2 = self.calc_partial_pressure_gas('n2')
        self.total_partials = self.calc_tot_partial_pressures()
        self.total_mol_mins = \
            self.n2_mol_min / self.partial_pressures_n2 * self.total_partials
        self.h2o_mol_mins = \
            (self.partial_pressures_h2o / self.total_partials) * \
            self.total_mol_mins
        self.h2o_flows = self.h2o_mol_mins * self.c_dict['molmass_h2o'] * 60


class MeasurementPoints(HumidityCalc):
    def __init__(self, currents, stoichiometries, humidities):
        super().__init__()
        self.data = {}
        self.currents = currents
        self.columns = \
            ['Current [A]', 'H2 [Nml/min]', 'H2O An [g/h]', 'Air [Nml/min]',
             'H2O Ca [g/h]', 'Stoichiometry An', 'Stoichiometry Ca',
             'Rel.Humidity An', 'Rel.Humidity Ca']
        self.data_update(currents, stoichiometries, humidities)

    def data_update(self, currents, stoichiometries, humidities):
        for cur in currents:
            self.stack_verbrauch = self.calc_stack_verbrauch(cur)
            for stoi in stoichiometries:
                air_flow = self.calc_flows_air(stoi[1])
                h2_flow = self.calc_flows_h2(stoi[0])
                for hum in humidities:
                    self.calc_flows_h2o(hum)
                    case = f'S_A{stoi[0]}_C{stoi[1]} H_A{hum[0]}_C{hum[1]}'
                    if case in self.data:
                        self.data[case].append(
                            [cur, h2_flow, self.h2o_flows[0], air_flow,
                             self.h2o_flows[1], stoi[0], stoi[1], hum[0],
                             hum[1]])
                    else:
                        self.data.update(
                            {case: [[cur, h2_flow, self.h2o_flows[0],
                                     air_flow, self.h2o_flows[1],
                                     stoi[0], stoi[1], hum[0], hum[1]]]})

    def generate_table(self, fname='compiled_table'):
        pdf = matplotlib.backends.backend_pdf.PdfPages(f'{fname}.pdf')

        row = [x for x in range(len(self.currents))]
        for key, val in self.data.items():
            data_array = np.asarray(val)
            df = \
                pd.DataFrame(data_array,
                             columns=['Current[A]', 'H2[Nml/min]',
                                      'H2O An[g/h]', 'Air[Nml/min]',
                                      'H2O Ca[g/h]', 'Stoichiometry An',
                                      'Stoichiometry Ca', 'Rel.Humidity An',
                                      'Rel.Humidity Ca'])
            ax = plt.subplot(111, frame_on=False)  # no visible frame
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis

            table(ax, df, loc='center')  # where df is your data frame
            plt.rcParams.update({'font.size': 22})
            plt.savefig(f'{key}.png')
            fig = plt.figure(figsize=(10, 8), dpi=136)
            ax = fig.add_subplot()
            ax.set_axis_off()
            ax.table(cellText=val, rowLabels=row, colLabels=self.columns,
                     cellLoc='center', loc='upper left')
            fig.suptitle(f'{key}')
            pdf.savefig(fig)
        pdf.close()

    def generate_xlsx(self, fname='compiled_xls'):
        writer = pd.ExcelWriter(f'{fname}.xlsx', engine='xlsxwriter')
        for key, d in self.data.items():
            arr_d = np.asarray(d)
            df = pd.DataFrame(arr_d, columns=self.columns)
            df.to_excel(writer, sheet_name=f'{key}')
        writer.save()




