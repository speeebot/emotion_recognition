import argparse, os, sys
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from parse import parse
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

#-------------------------------input handling--------------------------------

def check_TestOrTrain(user_input) :
  valid_input = ['train', 'test']
  for val in valid_input:
    if user_input == val:
      return user_input
  raise argparse.ArgumentTypeError("%s is an invalid argument (must be either \"train\" or \"test\")" % user_input)

def check_data_directory_path(user_input):
  if os.path.isdir(user_input):
    return user_input
  else:
    raise argparse.ArgumentTypeError("%s is not a valid data directory" % user_input)

def check_data_type(user_input) :
  error_msg = ("%s is an invalid data type (pick one: 'EDA', 'DIA', 'volt', "
                "'sys', 'resp', 'pulse', 'mean', 'mmhg', 'all')" % user_input)
  valid_input = ['EDA', 'DIA', 'volt', 'sys', 'resp', 'pulse', 'mean', 'mmhg', 'all']
  for val in valid_input:
    if user_input == val:
      return user_input
  raise argparse.ArgumentTypeError(error_msg)

def grab_args():
  # Initiate the parser
  parser = argparse.ArgumentParser()

  # Add long and short argument
  parser.add_argument('testOrTrain', type=check_TestOrTrain)
  parser.add_argument('data_directory_path', type=check_data_directory_path)
  parser.add_argument('model_name', type=str)
  parser.add_argument('data_type', type=check_data_type)

  # Read arguments from the command line
  args = parser.parse_args()
  return args

#-------------------------------data handling--------------------------------

def fix_data_type(data_type):
  if data_type == "EDA":
    return "EDA"
  elif data_type == "DIA":
    return "BP Dia"
  elif data_type == "volt":
    return "Resp"
  elif data_type == "sys":
    return "LA Systolic BP"
  elif data_type == "resp":
    return "Respiration Rate"
  elif data_type == "pulse":
    return "Pulse Rate"
  elif data_type == "mean":
    return "LA Mean BP"
  elif data_type == "mmhg":
    return "BP"
  else:
    return "all"
  
def normalize_data(data):
  min_val = np.min(data)
  max_val = np.max(data)
  if ((max_val - min_val) > 0):
    return (data - min_val) / (max_val - min_val)
  else:
    return data

def GetData(args, timestep, signal_count):
  data_type = args.data_type
  data_path = args.data_directory_path
  #map user inputs to their file name equivalents
  data_type = fix_data_type(data_type)
  #parse file name pattern
  pattern = '{subject_id}_{class_num}_{file_data_type}_{measurement_unit}.txt'

  #set data path based on training directory or testing directory
  if args.testOrTrain == "train":
    validation_data_path = data_path + "/Validation/"
    data_path += "/Training/"
  elif args.testOrTrain == "test":
    data_path += "/Testing/"

  #read training or testing data from directory and save it into an array
  arr_x = []
  arr_y = []
  with os.scandir(data_path) as it:
    for entry in it:
      if entry.name.endswith(".txt") and entry.is_file():
        result = entry.name.split("_")
        if result[2] == data_type and data_type != "all":
          #store the class type and data for this iteration of data_type file
          signals = np.loadtxt(data_path+entry.name)
          signals = list(signals)
          #normalize data
          signals = normalize_data(signals)
          arr_x.append(signals)
          label = int(result[1])
          #decrement labels by one, to fit between 0-9 for the output layer of 10 classes
          arr_y.append(label-1)
        elif data_type == "all":
          #store the class type and data for all data_type files
          signals = np.loadtxt(data_path+entry.name)
          signals = list(signals)
          signals = normalize_data(signals)
          arr_x.append(signals)
          label = int(result[1])
          arr_y.append(label-1)

  #read validation data from directory and save it into an array
  if args.testOrTrain == "train":
    arr_valid_x = []
    arr_valid_y = []
    with os.scandir(validation_data_path) as it:
      for entry in it:
        if entry.name.endswith(".txt") and entry.is_file():
          result = entry.name.split("_")
          if result[2] == data_type and data_type != "all":
            #store the class type and data for this iteration of data_type file
            signals = np.loadtxt(validation_data_path+entry.name)
            signals = list(signals)
            signals = normalize_data(signals)
            arr_valid_x.append(signals)
            label = int(result[1])
            arr_valid_y.append(label-1)
          elif data_type == "all":
            #store the class type and data for all data_type files
            signals = np.loadtxt(validation_data_path+entry.name)
            signals = list(signals)
            signals = normalize_data(signals)
            arr_valid_x.append(signals)
            label = int(result[1])
            arr_valid_y.append(label-1)

  #finishing preprocessing data, normalization
  if args.testOrTrain == "train":
    #format data
    x_train = np.array(arr_x, dtype=object)
    y_train = np.array(arr_y, dtype='float64')
    x_valid = np.array(arr_valid_x, dtype=object)
    y_valid = np.array(arr_valid_y, dtype='float64')
    #pad training data with 0s to make sequences same length
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=timestep, dtype='float32')
    x_valid = tf.keras.preprocessing.sequence.pad_sequences(x_valid, maxlen=timestep, dtype='float32')
    #reshape data for model
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_valid = np.reshape(x_valid, (x_valid.shape[0], 1, x_valid.shape[1]))

    return x_train, y_train, x_valid, y_valid
  elif args.testOrTrain == "test":
    #format data
    x_test = np.array(arr_x, dtype=object)
    y_test = np.array(arr_y, dtype='float64')
    #pad testing data with 0s to make sequences same length
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=timestep, dtype='float32')
    #reshape data for model
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    return x_test, y_test
  else:
    return x_train, y_train, x_test, y_test