from pathlib import Path

import pandas

import timeit
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.layers import Input, Dense, ELU

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import CSVLogger

import os
from tensorflow.keras.callbacks import CSVLogger

HISTORY_DIR = Path(__file__).resolve().parent / "History"
HISTORY_DIR.mkdir(exist_ok=True)

#-INITIAL SETTINGS------------------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Input" / "CSV"
data_name = 'B_SS'
note = " - training complete"

non = 72 # Number of nodes per hidden layer
nohl =  3  # Number of hidden layer

save_model = 'y'
save_history = 'y'
save_fig_acc = 'y'
save_fig_loss = 'y'

save_model_folder = 'Model'
save_his_folder = 'History'
save_fig_folder = 'Converging_Figure'

plot_acc = 'y'
plot_loss = 'y' 

#-DATA PROCESSING-------------------------------------------------------------------------------------------------------
X = pandas.read_csv(DATA_DIR / "Input_Python.csv").values
N = X.shape[0]

Y = pandas.read_csv(DATA_DIR / "Output_Python.csv").values
X = X[:N,1:]
Y = Y[:N,1:]

Ntrain =  int(0.9 * N)

x = X[:Ntrain, :] # Training samples - input
x_test = X[Ntrain:, :] # Validation samples - input

y = Y[:Ntrain, :] # Training samples - output
y_test = Y[Ntrain:, :] # Validation samples - output

#-DIMENSION OF THE INPUT AND OUTPUT DATA--------------------------------------------------------------------------------
no_input_node = x.shape[1]  # Number of input node
no_output_node = y.shape[1]  # Number of output node

#-NN STRUCTURE---------------------------------------------------------------------------------------------------------
model = Sequential()

#-Input layer---------------------------------
model.add(Input(shape=(no_input_node,)))

#-Hidden layer(s)-----------------------------
for i in range(nohl):
    model.add(Dense(non)) ; model.add(ELU(alpha=0.5))

#-Output layer--------------------------------
model.add(Dense(no_output_node))

#-TRAINING--------------------------------------------------------------------------------------------------------------
# Optimizing algorithm
optimizer = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adamax")

# Starting point of time of the training process
start = timeit.default_timer()

# Compile and Fit model
earlystopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0, patience=1000,restore_best_weights=True)

if save_history == 'y':
    csv_logger = CSVLogger(str(HISTORY_DIR / f"{data_name}.csv"), append=False)
    model.compile(loss=MeanSquaredError(), optimizer=optimizer, metrics=[tf.keras.metrics.MeanAbsoluteError()])
    training = model.fit(x, y, epochs=1500, batch_size=1, validation_data=(x_test, y_test), callbacks=[earlystopping, csv_logger])
else:
    model.compile(loss=MeanSquaredError(), optimizer=optimizer, metrics=[tf.keras.metrics.MeanAbsoluteError()])
    training = model.fit(x, y, epochs=1500, batch_size=1, validation_data=(x_test, y_test), callbacks=[earlystopping])

# Ending point of time of the training process
stop = timeit.default_timer()

#-SUMMARIZE THE PARAMETERS OF THE MODEL---------------------------------------------------------------------------------
print(model.summary())
print('')

#-EVALUATIONS-----------------------------------------------------------------------------------------------------------
evaluation = model.evaluate(x, y) # [loss, accuracy]
print('Loss_' + data_name + '    :', evaluation)
evaluation_test = model.evaluate(x_test, y_test)
print('Val_loss_' + data_name + ':', evaluation_test)
print('')

print(len(training.history['loss']), ' , ', (stop - start), '(s)') 
print('')

# Plot Acc and Val_acc values - Convergence investigating
if plot_acc == 'y' or save_fig_acc == 'y':
    acc_history = np.array(training.history['mean_absolute_error'])
    acc = acc_history[1:]
    val_acc_history = np.array(training.history['val_mean_absolute_error'])
    val_acc = val_acc_history[:-1]

    acc_fig = plt.figure('Acc_' + data_name)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('MAE versus Val_MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    if save_fig_acc == 'y':
       acc_fig.savefig(ROOT / "Converging_Figure" / "B_SS (MAE).pdf")

# Plot Loss and Val_loss values - Convergence investigating
if plot_loss == 'y' or save_fig_loss == 'y':
    loss_history = np.array(training.history['loss'])
    loss = loss_history[1:]
    val_loss_history = np.array(training.history['val_loss'])
    val_loss = val_loss_history[:-1]

    loss_fig = plt.figure('Loss_' + data_name)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Loss versus Val_Loss value')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    if save_fig_loss == 'y':
       loss_fig.savefig(ROOT / "Converging_Figure" / "B_SS (Loss).pdf")

#-SAVE THE TRAINED MODEL------------------------------------------------------------------------------------------------
if save_model == 'y':
    model.save(ROOT / "Model" / "B_SS.keras", overwrite=True)
        
    print('Model of ' + data_name + note + ' has been saved!')
    print('')
else:
    print('Model of ' + data_name + note + ' has NOT been saved!')
    print('')

print(np.isnan(X).sum())
print(np.isnan(Y).sum()) 

#-SHOW FIGURES----------------------------------------------------------------------------------------------------------
if plot_acc == 'y' or plot_loss == 'y':
    plt.show()