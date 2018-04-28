class Battery(object):
    """ Used to store information about the battery.

       :param current_charge: is the initial state of charge of the battery
       :param capacity: is the battery capacity in Wh
       :param charging_power_limit: the limit of the power that can charge the battery in W
       :param discharging_power_limit: the limit of the power that can discharge the battery in W
       :param battery_charging_efficiency: The efficiecny of the battery when charging
       :param battery_discharing_efficiecny: The discharging efficiency
    """
    def __init__(self,
                 capacity=0.0,
                 charging_power_limit=1.0,
                 discharging_power_limit=-1.0,
                 charging_efficiency=0.95,
                 discharging_efficiency=0.95):
        self.capacity = capacity
        self.charging_power_limit = charging_power_limit
        self.discharging_power_limit = discharging_power_limit
        self.charging_efficiency = charging_efficiency
        self.discharging_efficiency = discharging_efficiency

        # delta in the battery energy resulting from each action
        self.delta_energy = {}
        self.delta_energy['idle'] = 0
        self.delta_energy['charge'] = self.charging_power_limit * self.charging_efficiency / 4.0
        self.delta_energy['discharge'] = self.discharging_power_limit / self.discharging_efficiency / 4.0
        for quarter in range(1, 4):
            self.delta_energy['charge_'+str(quarter)] = quarter / 4.0 * self.delta_energy['charge']
            self.delta_energy['discharge_'+str(quarter)] = quarter / 4.0 * self.delta_energy['discharge']

        # by which factor we need to multiply delta_energy to get delta_grid_energy
        self.delta_grid_factor = {}
        self.delta_grid_factor['idle'] = 1
        self.delta_grid_factor['charge'] = 1 / self.charging_efficiency
        self.delta_grid_factor['discharge'] = 1 * self.discharging_efficiency
        for quarter in range(1, 4):
            self.delta_grid_factor['charge_'+str(quarter)] = self.delta_grid_factor['charge']
            self.delta_grid_factor['discharge_'+str(quarter)] = self.delta_grid_factor['discharge']
