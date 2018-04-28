from keras import backend as K


def custom_loss(y, output):
    # get auxiliary variables
    output_id = output[0, -1]  # identifies the current output
    yfac = y[:, -2]
    ycharge = y[:, -1]

    # now restrict to normal data
    output = output[:, :-1]
    y = y[:, :-2]

    # MSE
    MSE = (output - y) * (output - y)
    MSE = K.mean(MSE, axis=-1)
    MSE = K.mean(MSE)

    # MAP
    MAP = K.abs(output - y)
    MAP = K.mean(MAP, axis=-1)
    MAP = MAP / K.maximum(K.max(K.abs(output), axis=-1), K.max(K.abs(y), axis=-1))  # normalized
    MAP = K.mean(MAP)

    # score fair
    score = -K.max(y, axis=-1) * yfac
    score_fair1 = 5. * score * K.cast(K.less(ycharge, 1/5.), 'float32')
    score_fair2 = 10. * score * K.cast(K.less(ycharge, 1/10.), 'float32')

    # we must subtract twice the bias between score_fair1 and score_fair2
    score_fair = score_fair1 - 2 * (score_fair1 - score_fair2)

    is_MSE = K.cast(K.equal(output_id, 0), 'float32')
    is_MAP = K.cast(K.equal(output_id, 1), 'float32')
    is_score_fair = K.cast(K.equal(output_id, 2), 'float32')

    return MSE * is_MSE + MAP * is_MAP + score_fair * is_score_fair
