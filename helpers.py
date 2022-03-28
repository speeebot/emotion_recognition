import argparse, os, sys
from sklearn.metrics import precision_score, recall_score, confusion_matrix

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

  #-------------------------------save/load module------------------------------

