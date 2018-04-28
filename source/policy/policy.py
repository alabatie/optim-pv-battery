import os
import multiprocessing as mp

import numpy as np

from keras.models import Model

from keras.layers import Input, Dense, Lambda
from keras.layers import concatenate, TimeDistributed
from keras.layers import Activation

from keras.optimizers import adam
from keras.callbacks import ReduceLROnPlateau

from keras import backend as K


from policy.constants import NUM_THREADS, NUM_POLICY, LEN_POLICY, FAC_THRESHOLD_TRAIN, DELTA_DENSE, FINAL_TRAINING
from policy.constants import BATCH_SIZE, NUM_EPOCHS, NUM_LAYERS1, NUM_LAYERS2, NUM_UNITS, NUM_GENERATOR, PATIENCE_LR

from policy.keras_processing import simulation_function, custom_loss


learning_rate_reduction = ReduceLROnPlateau(monitor='loss',
                                            patience=PATIENCE_LR,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


class Policy(object):
    batch_size = BATCH_SIZE
    num_epochs = NUM_EPOCHS

    num_layers1 = NUM_LAYERS1
    num_layers2 = NUM_LAYERS2
    num_units = NUM_UNITS

    num_generator = NUM_GENERATOR
    model_output_directory = '../output/policy/models'

    def __init__(self, simu, batt, norm):
        # length of policy, number of policy
        self.len_policy = LEN_POLICY
        self.num_policy = NUM_POLICY
        self.len_total = self.num_policy * self.len_policy
        self.delta_dense = DELTA_DENSE

        # simu
        self.simu = simu
        self.len_simu = 959

        # maximum factor for normalizing score
        self.factor_max = {"train": FAC_THRESHOLD_TRAIN * simu["train"].factor_history,
                           "val": 4 * FAC_THRESHOLD_TRAIN * simu["train"].factor_history}

        # proba of each simu in each generator
        self.simu_prop_train = simu["train"].nt / float(simu["train"].nt + simu["submit"].nt)
        self.simu_list = ["train", "submit"]
        self.simu_proba = [self.simu_prop_train, 1-self.simu_prop_train]

        self.final_training = FINAL_TRAINING  # if final training, we use self.simu_proba for simu selection

        # maximum input charge
        self.input_charge_max = {"train": 1.0, "val": 0.0}

        # dimension of inputs
        self.nfeat_time = simu["train"].X_time[1].shape[1]
        self.nfeat_histo = simu["train"].X_histo[1].shape[1]
        self.nfeat_future = simu["train"].X_future[1].shape[1]
        self.nfeat_timestep = self.len_policy // self.delta_dense

        # optimization variables
        self.num_steps = int(self.num_generator / self.batch_size)

        # init shared layers
        self.init_layers()

        # create model
        self.model = self.make_model()

        # input timestep
        # the choice of triangular matrix is useful to keep the order relationship,
        # while allowing the model to consider transitions independently from each other
        # (i.e. to potentially give different importance to different transitions)
        X_timestep = np.tri(self.len_policy, self.len_policy)
        X_timestep = X_timestep.transpose()
        X_timestep = np.tile(X_timestep, (self.num_policy, 1))
        self.X_timestep = X_timestep[::self.delta_dense, ::self.delta_dense]

        # input battery (independent of batch sample, and timestep)
        self.X_battery = {}
        for batt_id in batt:
            self.X_battery[batt_id] = np.hstack((batt[batt_id].charging_efficiency, batt[batt_id].capacity,
                                                 batt[batt_id].charging_power_limit, norm["battery"]))

    def init_layers(self):
        self.dense_layer = {}
        for ipolicy in range(self.num_policy):
            self.dense_layer[ipolicy] = {}
            for ilayer in range(self.num_layers1):
                self.dense_layer[ipolicy][ilayer] = Dense(self.num_units, activation='selu')
            for ilayer in range(self.num_layers1, self.num_layers1 + self.num_layers2):
                self.dense_layer[ipolicy][ilayer] = Dense(self.num_units, activation='relu')

            self.dense_layer[ipolicy]['sigmoid'] = Dense(3*self.delta_dense)
            self.dense_layer[ipolicy]['softmax'] = Dense(3*self.delta_dense)

    def make_model(self):
        input_time = Input(shape=(self.len_total//self.delta_dense, self.nfeat_time), name='input_time')
        input_histo = Input(shape=(self.len_total//self.delta_dense, self.nfeat_histo), name='input_histo')
        input_future = Input(shape=(self.len_total//self.delta_dense, self.nfeat_future), name='input_future')
        input_timestep = Input(shape=(self.len_total//self.delta_dense, self.nfeat_timestep), name='input_timestep')
        input_simulation = Input(shape=(self.len_total, 4), name='input_simulation')

        input_charge = Input(shape=(1,), name='input_charge')  # only 1 input at first time step
        input_battery = Input(shape=(4,), name='input_battery')  # efficiency, capacity, power, normalization

        input_battery_norm = Lambda(lambda x: K.concatenate([x[:, :1], x[:, 1:2]/x[:, -1:], x[:, 2:3]/x[:, -1:]],
                                                            axis=-1))(input_battery)  # normalize
        input_battery_simulation = Lambda(lambda x: x[:, :-1])(input_battery)  # remove normalization factor

        input_policy_series = concatenate([input_histo, input_future], axis=-1)
        input_policy_other = concatenate([input_time, input_timestep], axis=-1)

        money_saved = []  # will store money_saved for all time steps
        money_saved_fair = []  # same, with just a different id
        sigmoid_output = []  # for visualizing % of activation
        efficiency_output = []  # for visualizing battery efficiency

        current_charge = input_charge
        for timestep in range(self.len_total):
            if timestep % self.delta_dense == 0:
                input_policy_series_timestep = Lambda(lambda x: x[:, timestep//self.delta_dense, :])(input_policy_series)
                input_policy_other_timestep = Lambda(lambda x: x[:, timestep//self.delta_dense, :])(input_policy_other)

                ipolicy = timestep // self.len_policy
                for ilayer in range(self.num_layers1):
                    input_policy_series_timestep = self.dense_layer[ipolicy][ilayer](input_policy_series_timestep)

                proposed_charge = concatenate([current_charge, input_battery_norm,
                                               input_policy_series_timestep, input_policy_other_timestep], axis=-1)

                for ilayer in range(self.num_layers1, self.num_layers1 + self.num_layers2):
                    proposed_charge = self.dense_layer[ipolicy][ilayer](proposed_charge)

                sigmoid_proposed_charge = self.dense_layer[ipolicy]['sigmoid'](proposed_charge)
                softmax_proposed_charge = self.dense_layer[ipolicy]['softmax'](proposed_charge)

                sigmoid_proposed_charge = Lambda(lambda x: K.reshape(x, (-1, self.delta_dense, 3)))(sigmoid_proposed_charge)
                sigmoid_proposed_charge = TimeDistributed(Activation('sigmoid'))(sigmoid_proposed_charge)

                softmax_proposed_charge = Lambda(lambda x: K.reshape(x, (-1, self.delta_dense, 3)))(softmax_proposed_charge)
                softmax_proposed_charge = TimeDistributed(Activation('softmax'))(softmax_proposed_charge)

            idense = timestep % self.delta_dense
            sigmoid_proposed_charge_timestep = Lambda(lambda x: x[:, idense, :])(sigmoid_proposed_charge)
            softmax_proposed_charge_timestep = Lambda(lambda x: x[:, idense, :])(softmax_proposed_charge)

            # get money saved, efficiency and current charge
            sigmoid_output.append(Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(sigmoid_proposed_charge_timestep))
            efficiency_output.append(Lambda(lambda x: x[:, :1])(input_battery))

            input_simulation_timestep = Lambda(lambda x: x[:, timestep, :])(input_simulation)
            input_simulation_timestep = concatenate([current_charge,
                                                     sigmoid_proposed_charge_timestep,
                                                     softmax_proposed_charge_timestep,
                                                     input_battery_simulation, input_simulation_timestep], axis=-1)

            output_simulation_timestep = Lambda(lambda x: simulation_function(x))(input_simulation_timestep)
            money_saved.append(Lambda(lambda x: x[:, :1])(output_simulation_timestep))
            money_saved_fair.append(Lambda(lambda x: x[:, :1])(output_simulation_timestep))
            current_charge = Lambda(lambda x: x[:, 1:])(output_simulation_timestep)

        # add id's to distinguish different outputs
        money_saved += [Lambda(lambda x: 0+0*x)(money_saved[0])]
        money_saved_fair += [Lambda(lambda x: 1+0*x)(money_saved_fair[0])]
        sigmoid_output += [Lambda(lambda x: 2+0*x)(sigmoid_output[0])]
        efficiency_output += [Lambda(lambda x: 2+0*x)(efficiency_output[0])]

        # concatenate all timesteps into same tensor
        money_saved = concatenate(money_saved, axis=-1, name="score")
        money_saved_fair = concatenate(money_saved_fair, axis=-1, name="score_fair")
        sigmoid_output = concatenate(sigmoid_output, axis=-1, name="sigmoid")
        efficiency_output = concatenate(efficiency_output, axis=-1, name="efficiency")

        # compile model
        model = Model(inputs=[input_time, input_histo, input_future, input_timestep,
                              input_charge, input_battery, input_simulation],
                      outputs=[money_saved, money_saved_fair, sigmoid_output, efficiency_output])
        model.compile(optimizer=adam(clipnorm=0.001, lr=0.001,  epsilon=1e-08, decay=0.0),
                      loss=custom_loss, loss_weights=[1., 0., 0., 0.])

        return model

    def data_generator(self, type_data):
        name_list = ['time', 'histo', 'future', 'timestep', 'charge', 'battery', 'simulation']

        type_simu_batch = np.random.choice(self.simu_list, p=self.simu_proba, size=self.batch_size) if self.final_training \
            else (["train"]*self.batch_size if (type_data == "train") else ["submit"]*self.batch_size)

        X = {name: [] for name in name_list}
        ynorm, ycharge = [], []
        for ibatch in range(self.batch_size):
            type_simu = type_simu_batch[ibatch]

            period_id = np.random.choice(self.simu[type_simu].period_list, p=self.simu[type_simu].proba_list)
            batt_id = np.random.choice([1, 2])

            len_period = self.simu[type_simu].len_period[period_id]

            start_load = np.random.randint(len_period-self.len_total+1)
            end_load = start_load + self.len_total

            if type_data == "train":
                # we choose the strongest data augmentation during training
                start_pv = np.random.randint(len_period-self.len_total+1)
                end_pv = start_pv + self.len_total
            elif type_data == "val":
                # during validation, we choose only nearly valid data augmentations
                start_pv_list = [start_load + 96*shift_pv for shift_pv in range(-14, 15)]
                start_pv_list = [start_pv for start_pv in start_pv_list
                                 if (start_pv >= 0) and (start_pv <= len_period-self.len_total)]
                start_pv = np.random.choice(start_pv_list)
                end_pv = start_pv + self.len_total

            # compute factor simu
            pv_simu = np.roll(self.simu[type_simu].X_actual[period_id][:, 0], -start_pv)[:self.len_simu]
            load_simu = np.roll(self.simu[type_simu].X_actual[period_id][:, 1], -start_load)[:self.len_simu]
            buy_price_simu = np.roll(self.simu[type_simu].X_actual[period_id][:, 2], -start_load)[:self.len_simu]
            sell_price_simu = np.roll(self.simu[type_simu].X_actual[period_id][:, 3], -start_load)[:self.len_simu]

            grid_energy_simu = load_simu - pv_simu
            price_simu = buy_price_simu * (grid_energy_simu >= 0) + sell_price_simu * (grid_energy_simu < 0)
            money_spent_simu = grid_energy_simu * (price_simu / 1000.)

            factor_simu = 1. / money_spent_simu.mean(keepdims=True)
            factor_simu = np.clip(factor_simu, -self.factor_max[type_data], self.factor_max[type_data])

            input_charge = self.input_charge_max[type_data] * np.random.random(1)

            X_histo_pv = self.simu[type_simu].X_histo[period_id][start_pv:end_pv:self.delta_dense, :self.nfeat_histo//2]
            X_histo_load = self.simu[type_simu].X_histo[period_id][start_load:end_load:self.delta_dense, self.nfeat_histo//2:]

            X_future_pv = self.simu[type_simu].X_future[period_id][start_pv:end_pv:self.delta_dense, :self.nfeat_future//4]
            X_future_load = self.simu[type_simu].X_future[period_id][start_load:end_load:self.delta_dense, self.nfeat_future//4:]

            X_actual_pv = self.simu[type_simu].X_actual[period_id][start_pv:end_pv, :1]
            X_actual_load = self.simu[type_simu].X_actual[period_id][start_load:end_load, 1:]

            X['time'].append(self.simu[type_simu].X_time[period_id][start_load:end_load:self.delta_dense, ])

            X['histo'].append(np.hstack((X_histo_pv, X_histo_load)))
            X['future'].append(np.hstack((X_future_pv, X_future_load)))
            X['simulation'].append(np.hstack((X_actual_pv, X_actual_load)))

            X['timestep'].append(self.X_timestep)
            X['charge'].append(input_charge)
            X['battery'].append(self.X_battery[batt_id])

            ynorm.append(1. / abs(factor_simu))
            ycharge.append(input_charge)

        # transform to numpy arrays
        for name in name_list:
            X[name] = np.array(X[name])
        y = np.hstack((np.array(ynorm), np.array(ycharge)))

        return [X[name] for name in name_list], [y, y, y, y]

    def data_generator_train(self):
        while True:
            yield self.data_generator("train")

    def data_generator_val(self):
        while True:
            yield self.data_generator("val")

    def train(self):
        self.iepoch = 0  # init epoch

        while self.iepoch < self.num_epochs:
            self.model.fit_generator(self.data_generator_train(), steps_per_epoch=self.num_steps,
                                     validation_data=self.data_generator_val(), validation_steps=self.num_steps//5,
                                     initial_epoch=self.iepoch, epochs=self.iepoch+1,
                                     verbose=True, callbacks=[learning_rate_reduction])

            self.iepoch += 1

    def make_deploy_model(self):
        input_time = Input(shape=(self.nfeat_time,))
        input_histo = Input(shape=(self.nfeat_histo,))
        input_future = Input(shape=(self.nfeat_future,))
        input_timestep = Input(shape=(self.nfeat_timestep,))

        input_charge = Input(shape=(1,))
        input_battery_norm = Input(shape=(3,))  # efficiency, capacity and power

        deploy_model = {}
        for ipolicy in range(self.num_policy):
            input_policy_series = concatenate([input_histo, input_future], axis=-1)
            input_policy_other = concatenate([input_time, input_timestep], axis=-1)

            for ilayer in range(self.num_layers1):
                input_policy_series = self.dense_layer[ipolicy][ilayer](input_policy_series)

            proposed_charge = concatenate([input_charge, input_battery_norm,
                                           input_policy_series, input_policy_other], axis=-1)

            for ilayer in range(self.num_layers1, self.num_layers1 + self.num_layers2):
                proposed_charge = self.dense_layer[ipolicy][ilayer](proposed_charge)

            sigmoid_proposed_charge = self.dense_layer[ipolicy]['sigmoid'](proposed_charge)
            softmax_proposed_charge = self.dense_layer[ipolicy]['softmax'](proposed_charge)

            deploy_model[ipolicy] = Model(inputs=[input_time, input_histo, input_future, input_timestep,
                                                  input_charge, input_battery_norm],
                                          outputs=[sigmoid_proposed_charge, softmax_proposed_charge])

            deploy_model[ipolicy].compile(optimizer=adam(), loss='mse')  # compile just to remove warning at loading

        return deploy_model

    def save(self, site_id):
        os.makedirs(self.model_output_directory, exist_ok=True)

        deploy_model = self.make_deploy_model()
        for ipolicy in range(self.num_policy):
            deploy_model[ipolicy].save(os.path.join(self.model_output_directory, f'{site_id}_{ipolicy}.h5'))
