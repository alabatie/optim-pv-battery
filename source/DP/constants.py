NUM_THREADS = 4
NUM_MODELS = 48  # number of successive models (with constant actions during 1h, this corresponds to 48 hours)
FAC_THRESHOLD_TRAIN = 5.0  # maximum factor allowed in score for individual simulation compared to whole history
FINAL_TRAINING = False   # whether we do final training, where we mix train/ and submit/

NUM_SIMU_BATCH = 2  # number of different simu periods in each batch
BATCH_SIZE_SIMU = 16  # number of points for each different simu period (BATCH_SIZE = product of these 2 values)
NUM_EPOCHS = 30  # number of epochs
NUM_LAYERS1 = 0  # number of layers processing only time series input
NUM_LAYERS2 = 3  # number of further layers
NUM_UNITS = 300  # number of units in each layer

NUM_GENERATOR = 100000  # number of samples seen during each epoch
PATIENCE_LR = 1  # patience to decrease the learning rate on plateau
