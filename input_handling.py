import argparse, os

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
    raise argparse.ArgumentTypeError("%s is not a valid directory" % user_input)

def check_data_type(user_input) :
  error_msg = ("%s is an invalid data type (pick one: 'EDA', 'DIA', 'volt', "
                "'sys', 'resp', 'pulse', 'mean', 'mmhg', 'all')" % user_input)
  valid_input = ['EDA', 'DIA', 'volt', 'sys', 'resp', 'pulse', 'mean', 'mmhg', 'all']
  for val in valid_input:
    if user_input == val:
      return user_input
  raise argparse.ArgumentTypeError(error_msg)