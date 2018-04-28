import os
import pickle

from calendar import monthrange
import numpy as np

from keras.models import load_model
import keras.backend as K


class BatteryContoller(object):
    """ The BatteryContoller class handles providing a new "target state of charge"
        at each time step.

        This class is instantiated by the simulation script, and it can
        be used to store any state that is needed for the call to
        propose_state_of_charge that happens in the simulation.

        The propose_state_of_charge method returns the state of
        charge between 0.0 and 1.0 to be attained at the end of the coming
        quarter, i.e., at time t+15 minutes.

        The arguments to propose_state_of_charge are as follows:
        :param site_id: The current site (building) id in case the model does different work per site
        :param timestamp: The current timestamp inlcuding time of day and date
        :param battery: The battery (see battery.py for useful properties, including current_charge and capacity)
        :param actual_previous_load: The actual load of the previous quarter.
        :param actual_previous_pv_production: The actual PV production of the previous quarter.
        :param price_buy: The price at which electricity can be bought from the grid for the
          next 96 quarters (i.e., an array of 96 values).
        :param price_sell: The price at which electricity can be sold to the grid for the
          next 96 quarters (i.e., an array of 96 values).
        :param load_forecast: The forecast of the load (consumption) established at time t for the next 96
          quarters (i.e., an array of 96 values).
        :param pv_forecast: The forecast of the PV production established at time t for the next
          96 quarters (i.e., an array of 96 values).

        :returns: proposed state of charge, a float between 0 (empty) and 1 (full).
    """

    def __init__(self):
        self.model = None
        self.norm = None

        self.sigmoid_proposed_charge = None
        self.softmax_proposed_charge = None

        self.histo_pv = []
        self.histo_load = []
        self.missing_value = -1.  # value for missing points in ov and load recent history

        self.num_policy = 12
        self.len_policy = 48
        self.delta_dense = 4

        # counter will determine which policy to call
        self.counter = 1  # start at 1 since simulation has only 959 timesteps

        self.num_policy_simu = 960 // self.len_policy
        self.ipolicy_list = [counter // self.len_policy for counter in range(960)]
        self.ipolicy_list = [max(ipolicy - self.num_policy_simu + self.num_policy, 0) for ipolicy in self.ipolicy_list]

        self.ipolicy_timestep_list = [counter % self.len_policy for counter in range(960)]
        self.idense_list = [ipolicy_timestep % self.delta_dense for ipolicy_timestep in self.ipolicy_timestep_list]
        for counter in range(1, self.delta_dense):  # call model at 1st time step and shift values for proposed_charge
            self.idense_list[counter] -= 1

        X_timestep = np.tri(self.len_policy, self.len_policy)
        X_timestep = X_timestep.transpose()
        self.X_timestep = X_timestep[::self.delta_dense, ::self.delta_dense]

    def _load_model(self, site_id):
        self.model = {}
        for ipolicy in range(self.num_policy):
            model_filename = os.path.join(self._get_directory(), f'{site_id}_{ipolicy}.h5')
            self.model[ipolicy] = load_model(model_filename)

    def _load_norms(self, site_id):
        norm_filename = os.path.join(self._get_directory(), f'norm_s{site_id}.p')
        with open(norm_filename, "rb") as file:
            self.norm = pickle.load(file)

    @staticmethod
    def _get_directory():
        return os.path.join('/simulation', 'simulate', 'assets') if os.path.isdir('/simulation') \
            else 'assets'

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def softmax(x):
        y = np.exp(x - np.max(x))
        y /= np.sum(y)
        return y

    @staticmethod
    def _get_proposed_charge(sigmoid_proposed_charge, softmax_proposed_charge, battery):
        if (isinstance(battery.current_charge, int) or isinstance(battery.current_charge, float)):
            current_charge = battery.current_charge
        else:
            current_charge = battery.current_charge[0][0]

        battery_efficiency = battery.charging_efficiency
        battery_capacity = battery.capacity
        battery_power = battery.charging_power_limit

        min_charge = max(current_charge - battery_power * (15. / 60.) / battery_efficiency / battery_capacity, 0.0)
        max_charge = min(current_charge + battery_power * (15. / 60.) * battery_efficiency / battery_capacity, 1.0)
        sigmoid_proposed_charge0 = min_charge + sigmoid_proposed_charge[0] * (max_charge - min_charge)

        min_charge1 = current_charge - battery_power * (15. / 60.) / battery_efficiency / battery_capacity
        max_charge1 = current_charge + battery_power * (15. / 60.) * battery_efficiency / battery_capacity
        sigmoid_proposed_charge1 = min_charge1 + sigmoid_proposed_charge[1] * (max_charge1 - min_charge1)
        sigmoid_proposed_charge1 = min(max(sigmoid_proposed_charge1, min_charge), max_charge)

        sigmoid_proposed_charge2 = min(max(sigmoid_proposed_charge[2], min_charge), max_charge)

        proposed_charge = softmax_proposed_charge[0] * sigmoid_proposed_charge0 \
            + softmax_proposed_charge[1] * sigmoid_proposed_charge1 \
            + softmax_proposed_charge[2] * sigmoid_proposed_charge2

        return proposed_charge

    @staticmethod
    def _resample_array(array, resample_fac):
        return np.array(array).reshape((-1, resample_fac)).mean(-1)

    def _downsample_future(self, data_future):
        return np.hstack((data_future[:16],
                          self._resample_array(data_future[16:48], 2),
                          self._resample_array(data_future[48:], 4)))

    def _downsample_histo(self, data_histo):
        data_histo = data_histo[::-1]
        return np.hstack((data_histo[:4],
                          self._resample_array(data_histo[4:16], 2),
                          self._resample_array(data_histo[16:96], 4),
                          self._resample_array(data_histo[96:7*96], 24)))

    def propose_state_of_charge(self,
                                site_id,
                                timestamp,
                                battery,
                                actual_previous_load,
                                actual_previous_pv_production,
                                price_buy,
                                price_sell,
                                load_forecast,
                                pv_forecast):
        # If needed, load model and norms
        if self.model is None:
            self._load_model(site_id)
            self._load_norms(site_id)

        # add previous pv and load to histo
        self.histo_pv.append(actual_previous_pv_production)
        self.histo_load.append(actual_previous_load)

        # get ipolicy,  ipolicy_timestep and idense
        ipolicy = self.ipolicy_list[self.counter]
        ipolicy_timestep = self.ipolicy_timestep_list[self.counter]
        idense = self.idense_list[self.counter]

        if idense == 0:
            # input time
            dt = timestamp.to_pydatetime()
            minute = dt.minute / 60.0
            hour = (dt.hour + minute) / 24.0
            day = (dt.day - 1 + hour) / float(monthrange(dt.year, dt.month)[1])
            weekday = (dt.weekday() + hour) / 7.0
            month = (dt.month - 1 + day) / 12.0

            input_time = np.array([[np.cos(2*np.pi * month), np.sin(2*np.pi * month),
                                    np.cos(2*np.pi * weekday), np.sin(2*np.pi * weekday),
                                    np.cos(2*np.pi * day), np.sin(2*np.pi * day),
                                    np.cos(2*np.pi * hour), np.sin(2*np.pi * hour),
                                    np.cos(2*np.pi * minute), np.sin(2*np.pi * minute)]])

            # input histo
            histo_pv = self.histo_pv[-7*96:]
            histo_load = self.histo_load[-7*96:]
            if len(histo_pv) != 7*96:
                histo_pv = np.array([np.nan] * (7*96-len(histo_pv)) + histo_pv)
                histo_load = np.array([np.nan] * (7*96-len(histo_load)) + histo_load)

            histo_pv = self._downsample_histo(histo_pv)
            histo_load = self._downsample_histo(histo_load)

            input_histo = np.hstack((histo_pv / self.norm["pv"], histo_load / self.norm["load"])).reshape((1, -1))
            input_histo[np.isnan(input_histo)] = self.missing_value   # set missing values

            # input future
            pv_forecast = self._downsample_future(pv_forecast)
            load_forecast = self._downsample_future(load_forecast)
            price_buy = self._downsample_future(price_buy)
            price_sell = self._downsample_future(price_sell)

            input_future = np.hstack((pv_forecast / self.norm["pv"],
                                      load_forecast / self.norm["load"],
                                      price_buy / self.norm["price"],
                                      price_sell / self.norm["price"])).reshape((1, -1))

            # input timestep
            input_timestep = self.X_timestep[[ipolicy_timestep // self.delta_dense], ]

            # input charge (battery.current_charge type varies...)
            if (isinstance(battery.current_charge, int) or isinstance(battery.current_charge, float)):
                input_charge = np.array([[battery.current_charge]])
            else:
                input_charge = battery.current_charge

            # input battery
            input_battery_norm = np.array([[
                battery.charging_efficiency,
                battery.capacity / self.norm["battery"],
                battery.charging_power_limit / self.norm["battery"]
            ]])

            self.sigmoid_proposed_charge, self.softmax_proposed_charge = \
                self.model[ipolicy].predict([input_time,
                                             input_histo,
                                             input_future,
                                             input_timestep,
                                             input_charge,
                                             input_battery_norm])

            self.sigmoid_proposed_charge = self.sigmoid_proposed_charge.reshape((self.delta_dense, 3))
            self.softmax_proposed_charge = self.softmax_proposed_charge.reshape((self.delta_dense, 3))

        sigmoid_proposed_charge = self.sigmoid(self.sigmoid_proposed_charge[idense, :])
        softmax_proposed_charge = self.softmax(self.softmax_proposed_charge[idense, :])

        proposed_charge = self._get_proposed_charge(sigmoid_proposed_charge, softmax_proposed_charge, battery)

        # update counter
        self.counter += 1

        if self.counter == 960:
            K.clear_session()

        # propose next battery charge
        return proposed_charge
