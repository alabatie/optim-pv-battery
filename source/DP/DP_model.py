import numpy as np
import multiprocessing as mp

from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, Lambda, concatenate
from keras import backend as K

from keras.optimizers import adam
from keras.callbacks import ReduceLROnPlateau

from DP.keras_processing import custom_loss

from DP.constants import NUM_MODELS, FAC_THRESHOLD_TRAIN, FINAL_TRAINING
from DP.constants import NUM_SIMU_BATCH, BATCH_SIZE_SIMU, NUM_EPOCHS, NUM_LAYERS1, NUM_LAYERS2, NUM_UNITS
from DP.constants import NUM_GENERATOR, PATIENCE_LR


learning_rate_reduction = ReduceLROnPlateau(monitor='loss',
                                            patience=PATIENCE_LR,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


class DP_model(object):
    num_simu_batch = NUM_SIMU_BATCH
    batch_size_simu = BATCH_SIZE_SIMU
    batch_size = NUM_SIMU_BATCH * BATCH_SIZE_SIMU
    num_epochs = NUM_EPOCHS

    num_layers1 = NUM_LAYERS1
    num_layers2 = NUM_LAYERS2
    num_units = NUM_UNITS

    # charge 100%, charge 25%, charge 50%, charge 75% - same for discharge
    list_actions = ['charge', 'charge_1', 'charge_2', 'charge_3',
                    'discharge', 'discharge_1', 'discharge_2', 'discharge_3', 'idle']
    num_actions = len(list_actions)

    num_generator = NUM_GENERATOR
    model_output_directory = '../output/DP/models'

    def __init__(self, imodel, simu, batt, norm, dp_next=None):
        # id model (just for printing)
        self.imodel = imodel
        self.len_Qvalue = NUM_MODELS - imodel  # number of timesteps until last DP_model

        # simu
        self.simu = simu
        self.len_simu = 959

        # maximum factor for normalizing score
        self.factor_max = FAC_THRESHOLD_TRAIN * simu["train"].factor_history

        # proba of each simu in each generator depending on FINAL_TRAINING
        self.simu_prop_train = simu["train"].nt / float(simu["train"].nt + simu["submit"].nt)
        self.simu_list = ["train", "submit"]
        self.simu_proba = [self.simu_prop_train, 1-self.simu_prop_train]

        self.final_training = FINAL_TRAINING  # if final training, we use self.simu_proba for simu selection

        # dimension of inputs
        self.nfeat_time = simu["train"].X_time[1].shape[1]
        self.nfeat_future = simu["train"].X_future[1].shape[1]
        self.nfeat_previous = simu["train"].X_previous[1].shape[1]

        # optimization variables
        self.num_steps = int(self.num_generator / self.batch_size)

        # create models
        self.modelD, self.modelD_deploy = {}, {}  # double networks (to deal with over-estimation bias)
        for iD in [1, 2]:
            self.modelD[iD], self.modelD_deploy[iD] = self.make_model()

        # battery and input battery
        self.batt = batt
        self.X_battery = {1: np.array([1, 0]), 2: np.array([0, 1])}

        # next DP model that will be called to get Q-value (None if it's the last DP_model)
        self.dp_next = dp_next

    def make_model(self):
        input_time = Input(shape=(self.nfeat_time,), name='input_time')
        input_future = Input(shape=(self.nfeat_future,), name='input_future')
        input_previous = Input(shape=(self.nfeat_previous,), name='input_previous')

        input_charge = Input(shape=(1,), name='input_charge')
        input_battery = Input(shape=(2,), name='input_battery')

        input_series = concatenate([input_future, input_previous], axis=-1)
        for ilayer in range(self.num_layers1):
            input_series = Dense(self.num_units)(input_series)
            input_series = Activation('relu')(input_series)

        # merge all inputs
        Qvalue = concatenate([input_charge, input_battery, input_time, input_series], axis=-1)
        for ilayer in range(self.num_layers2):
            Qvalue = Dense(self.num_units)(Qvalue)
            Qvalue = Activation('relu')(Qvalue)

        # last layer
        Qvalue = Dense(self.num_actions)(Qvalue)

        # add id's to distinguish different outputs
        Qvalue_MSE = Lambda(lambda x: K.concatenate([x, 0+0*x[:, :1]], axis=-1), name="Qvalue_MSE")(Qvalue)
        Qvalue_MAP = Lambda(lambda x: K.concatenate([x, 1+0*x[:, :1]], axis=-1), name="Qvalue_MAP")(Qvalue)
        score_fair = Lambda(lambda x: K.concatenate([x, 2+0*x[:, :1]], axis=-1), name="score_fair")(Qvalue)

        # make models
        model = Model(inputs=[input_time, input_future, input_previous, input_charge, input_battery],
                      outputs=[Qvalue_MSE, Qvalue_MAP, score_fair])

        # with MAE loss, output would approximate the median real Q-value
        # with MSE loss, output approximates the expected real Q-value (= what we want)
        model.compile(optimizer=adam(clipnorm=0.001, lr=0.001,  epsilon=1e-08, decay=0.0),
                      loss=custom_loss, loss_weights=[1., 0., 0.])

        # model for deployment
        model_deploy = Model(inputs=[input_time, input_future, input_previous, input_charge, input_battery],
                             outputs=[Qvalue])
        model_deploy.compile(optimizer=adam(), loss='mse')  # compile just to remove warning at loading

        return model, model_deploy

    def data_generator(self, type_data):
        name_list = ['time', 'future', 'previous', 'charge', 'battery']

        type_simu_batch = np.random.choice(self.simu_list, p=self.simu_proba, size=self.num_simu_batch) \
            if self.final_training \
            else (["train"]*self.num_simu_batch if (type_data == "train") else ["submit"]*self.num_simu_batch)

        X = {name: [] for name in name_list}
        yfac, ycharge = [], []
        reward = []
        X_next = {name: [] for name in name_list}
        for type_simu in type_simu_batch:
            period_id = np.random.choice(self.simu[type_simu].period_list, p=self.simu[type_simu].proba_list)
            len_period = self.simu[type_simu].len_period[period_id]

            batt_id = np.random.choice([1, 2])
            batt = self.batt[batt_id]

            start_load = np.random.randint(len_period - (self.batch_size_simu + 4) + 1)
            end_load = start_load + self.batch_size_simu + 4

            if type_data == "train":
                # we choose the strongest data augmentation during training
                start_pv = np.random.randint(len_period - (self.batch_size_simu + 4) + 1)
                end_pv = start_pv + self.batch_size_simu + 4
            elif type_data == "val":
                # during validation, we choose only nearly valid data augmentations
                start_pv_list = [start_load + 96*shift_pv for shift_pv in range(-14, 15)]
                start_pv_list = [start_pv for start_pv in start_pv_list
                                 if (start_pv >= 0) and (start_pv <= len_period - (self.batch_size_simu + 4))]
                start_pv = np.random.choice(start_pv_list)
                end_pv = start_pv + self.batch_size_simu + 4

            # compute factor simu
            pv_simu = np.roll(self.simu[type_simu].X_actual[period_id][:, 0], -start_pv)[:self.len_simu]
            load_simu = np.roll(self.simu[type_simu].X_actual[period_id][:, 1], -start_load)[:self.len_simu]
            buy_price_simu = np.roll(self.simu[type_simu].X_actual[period_id][:, 2], -start_load)[:self.len_simu]
            sell_price_simu = np.roll(self.simu[type_simu].X_actual[period_id][:, 3], -start_load)[:self.len_simu]

            grid_energy_simu = load_simu - pv_simu
            price_simu = buy_price_simu * (grid_energy_simu >= 0) + sell_price_simu * (grid_energy_simu < 0)
            money_spent_simu = grid_energy_simu * (price_simu / 1000.)

            factor_simu = 1. / abs(money_spent_simu.mean())
            factor_simu = np.clip(factor_simu, 0, self.factor_max)
            factor_simu *= self.len_Qvalue

            X_previous_pv = self.simu[type_simu].X_previous[period_id][start_pv:end_pv, :1]
            X_previous_load = self.simu[type_simu].X_previous[period_id][start_load:end_load, 1:]

            X_future_pv = self.simu[type_simu].X_future[period_id][start_pv:end_pv, :self.nfeat_future//4]
            X_future_load = self.simu[type_simu].X_future[period_id][start_load:end_load, self.nfeat_future//4:]

            X_time = self.simu[type_simu].X_time[period_id][start_load:end_load:, ]

            X_charge = np.random.random(self.batch_size_simu)
            X_battery = self.X_battery[batt_id]

            final_charge_action = np.zeros((self.batch_size_simu, self.num_actions))
            money_saved_action = np.zeros((self.batch_size_simu, self.num_actions))
            for iaction, action in enumerate(self.list_actions):
                current_energy = X_charge * batt.capacity
                money_saved = np.zeros(self.batch_size_simu)

                for delta_action in range(4):  # action is constant during 1 hour
                    delta_action_energy = current_energy
                    current_energy = delta_action_energy + batt.delta_energy[action]
                    current_energy = np.clip(current_energy, 0, batt.capacity)

                    delta_energy = current_energy - delta_action_energy
                    delta_grid_energy = delta_energy * batt.delta_grid_factor[action]

                    grid_energy = grid_energy_simu[delta_action:self.batch_size_simu+delta_action] + delta_grid_energy
                    price = buy_price_simu[delta_action:self.batch_size_simu+delta_action] * (grid_energy >= 0) \
                        + sell_price_simu[delta_action:self.batch_size_simu+delta_action] * (grid_energy < 0)
                    money_spent = grid_energy * (price / 1000.)
                    money_saved += (money_spent_simu[delta_action:self.batch_size_simu+delta_action] - money_spent)

                final_charge_action[:, iaction] = current_energy / batt.capacity
                money_saved_action[:, iaction] = money_saved

            X_next['charge'].append(final_charge_action)
            reward.append(money_saved_action)

            yfac.append(np.tile(factor_simu, self.batch_size_simu))
            ycharge.append(X_charge)

            X['previous'].append(np.hstack((X_previous_pv[:self.batch_size_simu, ],
                                            X_previous_load[:self.batch_size_simu, ])))
            X['future'].append(np.hstack((X_future_pv[:self.batch_size_simu, ],
                                          X_future_load[:self.batch_size_simu, ])))
            X['time'].append(X_time[:self.batch_size_simu, ])
            X['battery'].append(np.tile(X_battery, self.batch_size_simu))
            X['charge'].append(X_charge)

            X_next['previous'].append(np.hstack((X_previous_pv[4:self.batch_size_simu+4, ],
                                                 X_previous_load[4:self.batch_size_simu+4, ])))
            X_next['future'].append(np.hstack((X_future_pv[4:self.batch_size_simu+4, ],
                                               X_future_load[4:self.batch_size_simu+4, ])))
            X_next['time'].append(X_time[4:self.batch_size_simu+4, ])
            X_next['battery'].append(np.tile(X_battery, self.batch_size_simu))

        # transform reward, yfac, ycharge to numpy arrays
        reward = np.array(reward).reshape(-1)
        yfac = np.array(yfac).reshape((self.batch_size, -1))
        ycharge = np.array(ycharge).reshape((self.batch_size, -1))

        # transform X to numpy array
        for name in name_list:
            X[name] = np.array(X[name]).reshape((self.batch_size, -1))

        # transform X_next to numpy arrays
        X_next['charge'] = np.array(X_next['charge']).reshape((-1, 1))
        for name in ['time', 'future', 'previous', 'battery']:
            X_next[name] = np.array(X_next[name]).reshape((self.batch_size, -1))
            X_next[name] = np.tile(X_next[name], (self.num_actions, 1))

        # get Q-value from current timestep reward + Qvalue from next timestep
        y = reward + self.dp_next.get_max_Qvalue(X_next) if (self.dp_next is not None) else reward

        # reshape y with actions in axis = 1
        y = y.reshape((-1, self.num_actions))

        # add outputs for monitoring
        y = np.hstack((y, yfac, ycharge))

        return [X[name] for name in name_list], [y, y, y]

    def data_generator_train(self):
        while True:
            yield self.data_generator("train")

    def data_generator_val(self):
        while True:
            yield self.data_generator("val")

    def get_max_Qvalue(self, X_next):
        name_list = ['time', 'future', 'previous', 'charge', 'battery']

        Qvalue1 = self.modelD_deploy[1].predict([X_next[name] for name in name_list], batch_size=self.batch_size)
        Qvalue2 = self.modelD_deploy[2].predict([X_next[name] for name in name_list], batch_size=self.batch_size)

        indmax1 = np.argmax(Qvalue1, axis=1)
        indmax2 = np.argmax(Qvalue2, axis=1)

        # average results from 2 nets to be more accurate (each time, argmax and max are evaluated on separate nets)
        len_Q = len(indmax1)
        Qvalue = 0.5 * (Qvalue1[np.arange(len_Q), indmax2] + Qvalue2[np.arange(len_Q), indmax1])

        return Qvalue

    def train(self, verbose=True):
        for iD in [1, 2]:
            self.iepoch = 0   # init epoch

            while self.iepoch < self.num_epochs:
                # without workers = 0, it caused problems into TF since the generator calls another keras model
                self.modelD[iD].fit_generator(self.data_generator_train(), steps_per_epoch=self.num_steps,
                                              validation_data=self.data_generator_val(),
                                              validation_steps=self.num_steps//10,
                                              initial_epoch=self.iepoch, epochs=self.iepoch+1,
                                              verbose=verbose, workers=0,
                                              callbacks=[learning_rate_reduction])

                self.iepoch += 1

    def load(self, site_id):
        for iD in [1, 2]:
            self.modelD_deploy[iD] = load_model(os.path.join(self.model_output_directory, f'{site_id}_{self.imodel}_{iD}.h5'))

    def save(self, site_id):
        os.makedirs(self.model_output_directory, exist_ok=True)
        for iD in [1, 2]:
            self.modelD_deploy[iD].save(os.path.join(self.model_output_directory, f'{site_id}_{self.imodel}_{iD}.h5'))

    def clear(self):
        # free model memory
        self.modelD, self.modelD_deploy = {}, {}
        if self.dp_next is not None:
            self.dp_next.modelD, self.dp_next.modelD_deploy = {}, {}

        # clear backend session
        K.clear_session()
