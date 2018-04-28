NUM_THREADS = 4
NUM_POLICY = 8  # number of consecutive policies learned together
LEN_POLICY = 48  # duration of a single policy (the total duration of all consecutive policies is thus given by NUM_POLICY*LEN_POLICY)
FAC_THRESHOLD_TRAIN = 5.0  # maximum factor allowed in score for individual simulation compared to whole history
DELTA_DENSE = 4  # number of timesteps between 2 Dense layers
FINAL_TRAINING = False   # whether we do final training, where we mix train/ and submit/

BATCH_SIZE = 64  # batch size
NUM_EPOCHS = 40  # number of epochs
NUM_LAYERS1 = 0  # number of layers processing only time series input
NUM_LAYERS2 = 3  # number of further layers
NUM_UNITS = 300  # number of units in each layer

NUM_GENERATOR = 25000  # number of samples seen during each epoch
PATIENCE_LR = 2  # patience to decrease the learning rate on plateau
