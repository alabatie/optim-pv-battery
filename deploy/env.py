import os
from pathlib import Path

import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import pickle

from battery_controller import BatteryContoller
from battery import Battery
from simulate import Simulation


class PvBatteryEnv(object):
    nb_time_steps = 959

    battery_charge = None
    money_saved = None
    current_timestep_idx = None

    actual_previous_load = None
    actual_previous_pv = None

    simulation = None
    battery_controller = None

    site_id = None
    battery_id = None
    period_id = None

    def __init__(self, site_id=1, period_id=1, battery_id=1):
        """
        Resets the state of the environment and returns an initial observation.

        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed
                                  to be 0.
        """
        self.actual_previous_pv = []
        self.actual_previous_load = []
        self.battery_charge = []
        self.battery_energy = []
        self.grid_energy = []
        self.money_saved = []
        self.score = []
        self.current_timestep_idx = 0
        simulation_dir = (Path(__file__) / os.pardir / os.pardir).resolve()
        data_dir = simulation_dir / 'data'

        # load available metadata to determine the runs
        metadata_path = data_dir / 'metadata.csv'
        metadata = pd.read_csv(metadata_path, index_col=0)

        self.site_id = site_id
        self.period_id = period_id
        self.battery_id = battery_id

        site_data_path = data_dir / "submit" / f"{self.site_id}.csv"
        site_data = pd.read_csv(site_data_path, parse_dates=['timestamp'],
                                index_col='timestamp')

        parameters = metadata.loc[self.site_id]
        battery = Battery(capacity=parameters[f"Battery_{self.battery_id}_Capacity"] * 1000,
                          charging_power_limit=parameters[f"Battery_{self.battery_id}_Power"] * 1000,
                          discharging_power_limit=-parameters[f"Battery_{self.battery_id}_Power"] * 1000,
                          charging_efficiency=parameters[f"Battery_{self.battery_id}_Charge_Efficiency"],
                          discharging_efficiency=parameters[
                              f"Battery_{self.battery_id}_Discharge_Efficiency"])

        # n_periods = site_data.period_id.nunique() # keep as comment for now, might be useful later
        g_df = site_data[site_data.period_id == self.period_id]

        self.simulation = Simulation(g_df, battery, self.site_id)
        self.battery_controller = BatteryContoller()

    def step(self):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).

        # Returns
            observation (object): Agent's observation of the current environment.
            done (boolean): Whether the episode has ended, in which case further step() calls
                            will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging,
                         and sometimes learning).
        """

        money_saved, grid_energy = self.simulation.simulate_timestep(
            self.battery_controller,
            self.simulation.data.index[self.current_timestep_idx],
            self.simulation.data.iloc[self.current_timestep_idx])

        self.grid_energy.append(grid_energy)
        self.battery_charge.append(self.simulation.battery.current_charge)
        self.battery_energy.append(self.simulation.battery.capacity *
                                   self.simulation.battery.current_charge)
        self.money_saved.append(money_saved)
        self.score.append(self.simulation.score)
        self.actual_previous_load.append(self.simulation.actual_previous_load)
        self.actual_previous_pv.append(self.simulation.actual_previous_pv)

        self.current_timestep_idx += 1

    def render(self, mode='human', close=False):
        plt.subplot(2, 3, 1)
        plt.plot(self.battery_charge)
        plt.legend(['Charge', 'Target'])
        plt.title('Battery')

        plt.subplot(2, 3, 2)
        plt.plot(np.array(self.actual_previous_load) - np.array(self.actual_previous_pv))
        plt.title('Load - PV')

        plt.subplot(2, 3, 3)
        plt.plot(self.money_saved)
        plt.plot(np.cumsum(self.money_saved))
        plt.legend(['Instantaneous', 'Cumulative'])
        plt.title('Agregate reward')

        ax = plt.subplot(2, 3, 4)
        self.simulation.data[['price_sell_00', 'price_buy_00']].plot(ax=ax)

        plt.subplot(2, 3, 5)
        plt.plot(self.score)
        plt.title('Simulation score')

    def save(self):
        t = (self.battery_charge,
             self.battery_energy,
             self.simulation.data.actual_consumption,
             self.simulation.data.actual_pv,
             self.grid_energy,
             self.simulation.data['price_sell_00'],
             self.simulation.data['price_buy_00'],
             self.money_saved,
             self.score)

        # save data in pickle file
        simulation_dir = (Path(__file__) / os.pardir / os.pardir).resolve()
        pickle_path = simulation_dir/f"output/policy/env/env_s{self.site_id}_b{self.battery_id}_p{self.period_id}.p"
        with open(pickle_path, 'wb') as f:
            pickle.dump(t, f)


if __name__ == '__main__':
    site_id = 12
    period_id = 7
    battery_id = 2

    env = PvBatteryEnv(site_id=site_id, period_id=period_id, battery_id=battery_id)
    for _ in range(env.nb_time_steps):
        env.step()

    env.save()
    env.render()
    plt.show()
