from __future__ import print_function
import os 
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
# Used in visualization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.utils import plot_model   
# Utils
import submodules

from keras.callbacks import Callback
from datetime import datetime


start = datetime.now()



with open("start.txt", "w") as fp:
  fp.write("start!")
  fp.write(str(str))


# ==================================================== #
# Global Config
# ==================================================== #
MODELDIR_NAME = "models"
PARAM_DIR_NAME = 'results'
RESULTS_ROOTDIR_NAME = 'results'
MODEL_BASENAME = "model_"
PARAM_BASENAME = "param_"
MODEL_HDF5_NAME = "model.hdf5"


# ==================================================== #
# 1. Argment Parse
# ==================================================== #
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--conf', '-c', default="", help='/absolute/path/param_***.conf file path')
parser.add_argument('--model', '-m', default="", help='/absolute/path/model_***.py file path')
parser.add_argument('--logdir',  default="", help='/results/mode_***/param_***/ file path')
args = parser.parse_args()

# Setting File path
config_filepath = args.conf
_, param_file = args.conf.rsplit("/", 1)
PARAM_NAME = os.path.splitext(param_file)[0]

# Setting File path
_, model_file = args.model.rsplit("/", 1)
MODEL_NAME = os.path.splitext(model_file)[0]

# WORKINGLOG_DIR_PATH
WORKINGLOG_DIR_PATH = args.logdir

# ==================================================== #
# 2. Get Parameters (Config Parse)
# ==================================================== #
import configparser
config = configparser.ConfigParser()
config.read(config_filepath)

# Sction1
section1 = 'basic'
batch_size = config.getint(section1, 'batch_size')  # localhost
num_classes = config.getint(section1, 'num_classes')  # localhost
epochs = config.getint(section1, 'epochs')  # localhost


# ==================================================== #
# Setup 
# ==================================================== #


# ==================================================== #
# Training and Validation Datasets
# ==================================================== #
# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[:32,]
y_train = y_train[:32,]

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# ==================================================== #
#
# Model
#
# ==================================================== #
# Generate Model
model_module = submodules.load_model_from_modelpy(MODELDIR_NAME, MODEL_NAME)
model = model_module.create_model(x_train.shape[1:], num_classes)
model.summary() ## print model summary
with open(os.path.join(WORKINGLOG_DIR_PATH, "model.json") ,"w") as fp:
  fp.write(model.to_json())
plot_model(model, to_file=os.path.join(WORKINGLOG_DIR_PATH, 'model.png'), show_shapes=True)

# ==================================================== #
# Report Call Back
# ==================================================== #
# Call Back 1 : save model
cb_check = keras.callbacks.ModelCheckpoint(os.path.join(WORKINGLOG_DIR_PATH, MODEL_HDF5_NAME))

# Call Back 1 : Tensor Board
tensorboard_logs_dir_path = submodules.generate_dir([WORKINGLOG_DIR_PATH, "logs"])
cb_tensorboard = TensorBoard(  log_dir=tensorboard_logs_dir_path,
                            histogram_freq=0, 
                            write_graph=True)

# Call Back 1  CSV
cb_csvlogger = keras.callbacks.CSVLogger(
                              os.path.join(WORKINGLOG_DIR_PATH,"training.csv"), 
                              separator=',', 
                              append=False)

# ==================================================== #
# Training
# ==================================================== #
histroy = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=[cb_check, cb_tensorboard, cb_csvlogger])

# ==================================================== #
# Report
# ==================================================== #
score = model.evaluate(x_test, y_test, verbose=0)
## Print and visualize the results 
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = datetime.now()
elapsed = end-start

with open("success.txt", "w") as fp:
  fp.write("success!")
  fp.write(str(elapsed))

