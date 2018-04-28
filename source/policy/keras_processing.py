from keras import backend as K


def simulation_function(x):
    current_charge = x[:, 0]
    sigmoid_proposed_charge = x[:, 1:4]
    softmax_proposed_charge = x[:, 4:7]
    battery_efficiency = x[:, 7]
    battery_capacity = x[:, 8]
    battery_power = x[:, 9]
    actual_pv = x[:, 10]
    actual_consumption = x[:, 11]
    price_buy = x[:, 12]
    price_sell = x[:, 13]

    min_charge = K.maximum(current_charge - battery_power * (15. / 60.) / battery_efficiency / battery_capacity, 0.0)
    max_charge = K.minimum(current_charge + battery_power * (15. / 60.) * battery_efficiency / battery_capacity, 1.0)
    sigmoid_proposed_charge0 = min_charge + sigmoid_proposed_charge[:, 0] * (max_charge - min_charge)

    min_charge1 = current_charge - battery_power * (15. / 60.) / battery_efficiency / battery_capacity
    max_charge1 = current_charge + battery_power * (15. / 60.) * battery_efficiency / battery_capacity
    sigmoid_proposed_charge1 = min_charge1 + sigmoid_proposed_charge[:, 1] * (max_charge1 - min_charge1)
    sigmoid_proposed_charge1 = K.minimum(K.maximum(sigmoid_proposed_charge1, min_charge), max_charge)

    sigmoid_proposed_charge2 = K.minimum(K.maximum(sigmoid_proposed_charge[:, 2], min_charge), max_charge)

    proposed_charge = softmax_proposed_charge[:, 0] * sigmoid_proposed_charge0 \
        + softmax_proposed_charge[:, 1] * sigmoid_proposed_charge1 \
        + softmax_proposed_charge[:, 2] * sigmoid_proposed_charge2

    # calculate proposed energy change in the battery
    actual_energy_change = (proposed_charge - current_charge) * battery_capacity

    # update current charge
    current_charge = proposed_charge

    # efficiency is different whether we intend to charge or discharge
    actual_charging_power = K.relu(actual_energy_change) / battery_efficiency / (15. / 60.) \
        - K.relu(-actual_energy_change) * battery_efficiency / (15. / 60.)

    # what we need from the grid = the power put into the battery + the consumption - what is available from pv
    grid_energy = (actual_charging_power * (15. / 60.) + actual_consumption) - actual_pv

    # what we would need from the grid without battery
    grid_energy_without_battery = actual_consumption - actual_pv

    # compute spending for timestep
    money_spent = K.relu(grid_energy) * price_buy / 1000. \
        - K.relu(-grid_energy) * price_sell / 1000.

    # compute spending for timestep without battery
    money_spent_without_battery = K.relu(grid_energy_without_battery) * price_buy / 1000. \
        - K.relu(-grid_energy_without_battery) * price_sell / 1000.

    money_saved = money_spent - money_spent_without_battery

    return K.concatenate([K.expand_dims(money_saved, axis=-1), K.expand_dims(current_charge, axis=-1)])


def custom_loss(y, output):
    # get auxiliary variables
    output_id = output[0, -1]
    norm_score = y[:, 0]
    input_charge = y[:, 1]

    # now restrict to normal data
    output = output[:, :-1]

    # sigmoid or efficiency average
    output_sigmoid_efficiency = K.mean(output)

    # simulation score
    is_val = K.cast(K.equal(input_charge, 0.), 'float32')

    factor_fair = 2. * K.cast(K.less(input_charge, 0.5), 'float32')

    score = K.mean(output, axis=-1) / norm_score
    score_fair = score * factor_fair

    score_fair = K.mean(score_fair)
    score = K.mean(score)

    score_fair = score - 2 * (score-score_fair)  # remove twice the bias between score and score_fair
    score_fair = score * is_val + score_fair * (1-is_val)

    is_score = K.cast(K.equal(output_id, 0), 'float32')
    is_score_fair = K.cast(K.equal(output_id, 1), 'float32')
    is_sigmoid_efficiency = K.cast(K.equal(output_id, 2), 'float32')

    return score * is_score + score_fair * is_score_fair + output_sigmoid_efficiency * is_sigmoid_efficiency
