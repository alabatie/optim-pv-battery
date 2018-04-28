import multiprocessing as mp
from calendar import monthrange

import numpy as np

MISSING = -1.  # value for missing points in consumption recent history


def downsample_future(data_future):
    return np.concatenate((data_future[:, :, :16],
                           resample_array(data_future[:, :, 16:48], 2),
                           resample_array(data_future[:, :, 48:], 4)), axis=-1)


def downsample_histo(data_histo):
    data_histo = data_histo[:, :, ::-1]
    return np.concatenate((data_histo[:, :, :4],
                           resample_array(data_histo[:, :, 4:16], 2),
                           resample_array(data_histo[:, :, 16:96], 4),
                           resample_array(data_histo[:, :, 96:7*96], 24)), axis=-1)


def resample_array(array, resample_fac):
    return array.reshape((array.shape[0], array.shape[1], -1, resample_fac)).mean(-1)


def convert_2d(data_array):
    return data_array.reshape((data_array.shape[0], -1))


class Simulation(object):
    def __init__(self, data, norm, num_threads=1):
        """ Simulation object storing precomputed quantities used during training
        """

        self.data = data.copy()
        self.norm = norm

        # set number of threads
        self.num_threads = num_threads

        # initialization
        self.data['actual_previous_load'] = self.data['actual_consumption']
        self.data['actual_previous_pv'] = self.data['actual_pv']

        # align actual as the following, not the previous 15 minutes to
        # simplify simulation
        self.data.loc[:, 'actual_consumption'] = self.data.actual_consumption.shift(-1)
        self.data.loc[:, 'actual_pv'] = self.data.actual_pv.shift(-1)

        # period id and time
        self.period_id = self.data.period_id
        self.nt = len(self.data.index)

        # to sample periods
        self.len_period = {}
        self.period_list = []
        self.proba_list = []

        self.load_columns = self.data.columns.str.startswith('load_')
        self.pv_columns = self.data.columns.str.startswith('pv_')
        self.price_sell_columns = self.data.columns.str.startswith('price_sell_')
        self.price_buy_columns = self.data.columns.str.startswith('price_buy_')

        # dict storing precomputed inputs for policy models
        self.X_time = {}
        self.X_histo = {}
        self.X_future = {}
        self.X_actual = {}

        # to compute normalizing factor of each simu
        self.money_spent_without_battery = {}

    def precompute_thread(self, ithread, output_q):
        """Threaded Precomputations
        """
        X_time, X_histo, X_future, X_actual = {}, {}, {}, {}
        for timestep in range(self.nt)[ithread::self.num_threads]:
            # time variables
            dt = self.data.index[timestep]

            minute = dt.minute / 60.0
            hour = (dt.hour + minute) / 24.0
            day = (dt.day - 1 + hour) / float(monthrange(dt.year, dt.month)[1])
            weekday = (dt.weekday() + hour) / 7.0
            month = (dt.month - 1 + day) / 12.0

            X_time[timestep] = np.hstack((  # we introduce 2D representation to take into account the periodicity
                                            # e.g. we want the 1st of January to be close to the 31st of December
                                            # as a moment of the year
                                            np.cos(2*np.pi * month), np.sin(2*np.pi * month),
                                            np.cos(2*np.pi * weekday), np.sin(2*np.pi * weekday),
                                            np.cos(2*np.pi * day), np.sin(2*np.pi * day),
                                            np.cos(2*np.pi * hour), np.sin(2*np.pi * hour),
                                            np.cos(2*np.pi * minute), np.sin(2*np.pi * minute)))

            # now load, pv and price
            data_timestep = self.data.loc[dt]

            histo_load = self.data.actual_consumption[timestep-7*96:timestep]
            histo_pv = self.data.actual_pv[timestep-7*96:timestep]

            mask_invalid = (self.data.period_id[timestep-7*96:timestep] != self.data.period_id[timestep])
            histo_load[mask_invalid] = np.nan
            histo_pv[mask_invalid] = np.nan

            if len(histo_load) != 7*96:
                histo_load = np.hstack((np.tile(np.nan, 7*96-len(histo_load)), histo_load))
                histo_pv = np.hstack((np.tile(np.nan, 7*96-len(histo_pv)), histo_pv))

            X_histo[timestep] = np.vstack((histo_pv / self.norm["pv"],
                                           histo_load / self.norm["load"]))

            X_future[timestep] = np.vstack((data_timestep[self.pv_columns] / self.norm["pv"],
                                            data_timestep[self.load_columns] / self.norm["load"],
                                            data_timestep[self.price_buy_columns] / self.norm["price"],
                                            data_timestep[self.price_sell_columns] / self.norm["price"]))

            X_actual[timestep] = np.hstack((self.data.actual_pv[timestep],
                                            self.data.actual_consumption[timestep],
                                            self.data.price_buy_00[timestep],
                                            self.data.price_sell_00[timestep]))

        # aggregate all X's, and put in the queue
        X = (X_time, X_histo, X_future, X_actual)
        output_q.put(X)
        output_q.put(None)

    def precompute(self):
        """Precomputations
        """
        output_q = mp.Queue()

        # start workers
        job_list = [mp.Process(target=self.precompute_thread, args=(ithread, output_q,))
                    for ithread in range(self.num_threads)]
        for job in job_list:
            job.start()

        # get worker outputs
        X_time, X_histo, X_future, X_actual = {}, {}, {}, {}

        num_ended = 0
        while not num_ended == self.num_threads:
            X = output_q.get()
            if X is None:
                num_ended += 1
            else:
                X_time.update(X[0])
                X_histo.update(X[1])
                X_future.update(X[2])
                X_actual.update(X[3])

        # join workers
        for job in job_list:
            job.join()

        # prices and energy without battery
        buy_price = self.data.price_buy_00
        sell_price = self.data.price_sell_00
        grid_energy_without_battery = self.data.actual_consumption - self.data.actual_pv

        # money spent without battery
        price_without_battery = buy_price * (grid_energy_without_battery >= 0) \
            + sell_price * (grid_energy_without_battery < 0)
        money_spent_without_battery = grid_energy_without_battery * (price_without_battery / 1000.)

        # factor to normalize score computed on whole history
        self.factor_history = 1. / money_spent_without_battery.mean()

        # make period_id's as dictionary keys of all series (to easily select periods)
        for timestep in range(self.nt):
            period_id = self.period_id[timestep]
            if period_id not in self.X_time:
                self.X_time[period_id] = []
                self.X_histo[period_id] = []
                self.X_future[period_id] = []
                self.X_actual[period_id] = []

            self.X_time[period_id].append(X_time[timestep])
            self.X_histo[period_id].append(X_histo[timestep])
            self.X_future[period_id].append(X_future[timestep])
            self.X_actual[period_id].append(X_actual[timestep])

        # transform lists into numpy's (always exclude last time step of period with invalid actual consumption)
        for period_id in self.X_time:
            self.X_time[period_id] = np.array(self.X_time[period_id])[:-1, ]
            self.X_histo[period_id] = np.array(self.X_histo[period_id])[:-1, ]
            self.X_future[period_id] = np.array(self.X_future[period_id])[:-1, ]
            self.X_actual[period_id] = np.array(self.X_actual[period_id])[:-1, ]

            # downsample
            self.X_histo[period_id] = downsample_histo(self.X_histo[period_id])
            self.X_future[period_id] = downsample_future(self.X_future[period_id])

            # back to 2D
            self.X_histo[period_id] = convert_2d(self.X_histo[period_id])
            self.X_future[period_id] = convert_2d(self.X_future[period_id])

            # set missing values
            mask_missing = np.isnan(self.X_histo[period_id])
            self.X_histo[period_id][mask_missing] = MISSING

        # get proba for sampling periods
        self.len_period = {period_id: self.X_time[period_id].shape[0] for period_id in self.X_time}
        self.period_list = list(self.len_period.keys())
        self.proba_list = list(self.len_period.values())
        self.proba_list = [proba / float(sum(self.proba_list)) for proba in self.proba_list]
