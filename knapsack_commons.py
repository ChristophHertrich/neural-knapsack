import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import statsmodels.api as sm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_dense(n_neurons, activation, reg):
    return layers.Dense(n_neurons, activation=activation,
                        kernel_regularizer = reg,
                        bias_regularizer = reg)

def apply_concat_dense(network_inputs, prev_outputs, n_neurons, activation, reg):
    concat = layers.Concatenate()([network_inputs, prev_outputs])
    return get_dense(n_neurons, activation, reg)(concat)

def get_nn_model(layer_widths, reg):
    all_inputs = layers.Input(shape=(layer_widths[0],))
    x = get_dense(layer_widths[1], 'relu', reg)(all_inputs)
    for width in layer_widths[2:-1]:
        x = apply_concat_dense(all_inputs, x, width, 'relu', reg)
    x = apply_concat_dense(all_inputs, x, layer_widths[-1], None, reg)
    return tf.keras.Model(inputs=all_inputs, outputs=x)

def get_constant_width_model(pstar, width, reg):
    # input sequence: fin, pin, sin
    return get_nn_model([pstar+2, width, width, width, pstar], reg)

def get_theoretical_size_model(pstar, reg):
    # input sequence: fin, pin, sin
    return get_nn_model([pstar+2, 2*pstar, pstar*(pstar-1)/2, pstar, pstar], reg)

def get_random_knapsack_instance(pstar):
    remaining_profit = pstar
    profits = []
    while remaining_profit > 0:
        profit = np.random.randint(1, remaining_profit+1)
        remaining_profit -= profit
        profits.append(profit)
    profits = np.array(profits)
    np.random.shuffle(profits)
    sizes = np.random.random(len(profits))
    overload_factor = 1+np.random.random()
    sizes = sizes * overload_factor / sum(sizes)
    return profits, sizes

def dp_step(f, profit, size):
    pstar = len(f)
    f_new = np.zeros(pstar)
    for p in range(pstar):
        use_item = size
        if p >= profit:
            use_item += f[p-profit]
        f_new[p] = min(f[p], use_item)
    return f_new

def instance_to_training_data(profits, sizes, pstar):
    f_old = np.array([2.0] * pstar)
    training_data = []
    for profit, size in zip(profits, sizes):
        f_new = dp_step(f_old, profit, size)
        full_input = np.append(f_old, [profit, size])
        training_data.append((full_input, f_new))
        f_old = f_new
    return training_data

def evaluate_full_instance(pstar, model):
    profits, sizes = get_random_knapsack_instance(pstar)
    f_true = np.array([2.0] * pstar)
    f_model = np.array([2.0] * pstar)
    for profit, size in zip(profits, sizes):
        f_true = dp_step(f_true, profit, size)
        model_input = np.append(f_model, [profit, size])
        model_input = np.reshape(model_input, (1, len(model_input)))
        f_model = model(model_input)
    mse = tf.keras.losses.mse(f_true, f_model)
    return mse.numpy().item()

def evaluate_full_instances(pstar, model, n_instances):
    mses = []
    for _ in range(n_instances):
        mses.append(evaluate_full_instance(pstar, model))
    return sum(mses)/len(mses)

def prepare_dataset(pstar, batch_size, steps_per_epoch):
    def my_generator():
        while True:
            profits, sizes = get_random_knapsack_instance(pstar)
            current_data = instance_to_training_data(profits, sizes, pstar)
            for point in current_data:
                yield point
    dataset = tf.data.Dataset.from_generator(
        my_generator,
        (tf.float32, tf.float32),
        (tf.TensorShape([pstar+2]), tf.TensorShape([pstar])))
    dataset = dataset.shuffle(batch_size * steps_per_epoch)
    dataset = dataset.batch(batch_size)
    return dataset

def one_experiment(seed, pstar, width, reg, batch_size, steps_per_epoch, epochs, patience, verbose, eval_n):
    print('Current pstar = ' + str(pstar) + '. Current width = ' + str(width))
    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = get_constant_width_model(pstar, width, reg)
    model.compile(loss = 'mse', optimizer = 'adam')
    dataset = prepare_dataset(pstar, batch_size, steps_per_epoch)
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=patience, verbose=verbose)]
    history = model.fit(dataset, epochs = epochs, steps_per_epoch = steps_per_epoch,
              verbose = verbose, callbacks=callbacks)
    n_epochs = len(history.history['loss'])
    test_set = prepare_dataset(pstar, batch_size, steps_per_epoch)
    loss = model.evaluate(test_set, verbose = verbose, steps = steps_per_epoch)
    mse = evaluate_full_instances(pstar, model, eval_n)
    print('MSE: ' + str(mse) + '; loss: ' + str(loss) + ' after ' + str(n_epochs) + ' epochs')
    return {
        "pstar": pstar,
        "width": width,
        "mse": mse,
        "loss": loss,
        "n_epochs": n_epochs
    }

def read_results(filename):
    f = open(filename)
    results = json.load(f)
    f.close()
    return results

def perform_lin_reg(x_in, y_in, degree):
    x = np.array(x_in)
    if degree == 2:
        X = np.column_stack((x, x**2))
    elif degree ==1:
        X = x
    else:
        raise ValueError('Only degrees 1 and 2 are supported')
    X = sm.add_constant(X)
    y = np.array(y_in)
    model = sm.OLS(y, X)
    res = model.fit()
    print(res.summary())
    return res.params