import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from parse import parse
from helpers import *
np.set_printoptions(threshold=sys.maxsize)

#-----------------------Class types-----------------------------
#1: Happy, 2: Surprised, 3: Sad, 4: Startled/Surprised, 5: Skeptical
#6: Embarrassed, 7: Scared/Nervous, 8: Physical Pain, 9: Angry, 10: Disgusted

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
  return (data - np.min(data)) / (np.max(data) - np.min(data))


def GetData(args, timestep):
  data_type = args.data_type
  data_path = args.data_directory_path

  data_type = fix_data_type(data_type)

  pattern = '{subject_id}_{class_num}_{file_data_type}_{measurement_unit}.txt'

  #set data path based on user input
  if args.testOrTrain == "train":
    validation_data_path = data_path + "/Validation/"
    data_path += "/Training/"
  elif args.testOrTrain == "test":
    data_path += "/Testing/"

  file_count = len(os.listdir(data_path))

  #read training or testing data from directory
  arr = np.empty((0, 2))
  with os.scandir(data_path) as it:
    for entry in it:
      if entry.name.endswith(".txt") and entry.is_file():
        result = entry.name.split("_")
        if result[2] == data_type and data_type != "all":
          signals = np.loadtxt(data_path+entry.name)
          #store the class type and data for this iteration of data_type file
          arr = np.append(arr, np.array([[result[1], signals]], dtype=object), axis=0)
        elif data_type == "all":
          signals = np.loadtxt(data_path+entry.name)
          #store the class type and data for all data_type files
          arr = np.append(arr, np.array([[result[1], signals]], dtype=object), axis=0)

  #read validation data from directory
  if args.testOrTrain == "train":
    validation_arr = np.empty((0, 2))
    with os.scandir(validation_data_path) as it:
      for entry in it:
        if entry.name.endswith(".txt") and entry.is_file():
          result = entry.name.split("_")
          if result[2] == data_type and data_type != "all":
            signals = np.loadtxt(validation_data_path+entry.name)
            #store the class type and data for this iteration of data_type file
            validation_arr = np.append(validation_arr, np.array([[result[1], signals]], dtype=object), axis=0)
          elif data_type == "all":
            signals = np.loadtxt(validation_data_path+entry.name)
            #store the class type and data for all data_type files
            validation_arr = np.append(validation_arr, np.array([[result[1], signals]], dtype=object), axis=0)

  if args.testOrTrain == "train":
    x_train = arr[:, 1]
    y_train = arr[:, 0]
    x_test = validation_arr[:, 1]
    y_test = validation_arr[:, 0] 

    #pad training data with 0 (make sure same length)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=timestep, dtype='float32')
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=timestep, dtype='float32')

    y_train = np.array(y_train, dtype='float32')
    y_test = np.array(y_test, dtype='float32')

    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)

    return x_train, y_train, x_test, y_test
  elif args.testOrTrain == "test":
    x_test = arr[:, 1]
    y_test = arr[:, 0]
    #pad testing data with 0 (make sure same length)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=timestep, dtype='float32')
    y_test = np.array(y_test)   

    return x_test, y_test
  else:
    return x_train, y_train, x_test, y_test

def MakeModel(numFeatures, timestep, signal_count):
  #create sequential model
  model = tf.keras.models.Sequential()
  #create Embedding layer - takes input and makes it easy to use with LSTM
  #can "manually" do this yourself if you don't want to use embedding
  model.add(tf.keras.layers.Embedding(signal_count, numFeatures, input_length=timestep))
  #uncomment link below for vanilla RNN
  #model.add(tf.keras.layers.SimpleRNN(64))
  #uncomment line below for bidirectional LSTM
  #model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
  #uncomment line below to add another LSTM layer - return_sequences=True passes all hidden states to the next layer which is needed. Setting this to true is needed for all 
  #model.add(tf.keras.layers.LSTM(64, return_sequences=True))
  model.add(tf.keras.layers.LSTM(64))
  model.add(tf.keras.layers.Dense(128))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
  model.summary()
  
  return model

def Predict(model, x_test, y_test):
  #predict and format output to use with sklearn
  predict = model.predict(x_test)
  predict = np.argmax(predict, axis=1)
  #macro precision and recall
  precisionMacro = precision_score(y_test, predict, average='macro')
  recallMacro = recall_score(y_test, predict, average='macro')
  #micro precision and recall
  precisionMicro = precision_score(y_test, predict, average='micro')
  recallMicro = recall_score(y_test, predict, average='micro')
  confMat = confusion_matrix(y_test, predict)

  print("Macro precision: ", precisionMacro)
  print("Micro precision: ", precisionMicro)
  print("Macro recall: ", recallMacro)
  print("Micro recall: ", recallMicro)
  print(confMat)

def Train(args, numFeatures, timestep, signal_count):
  x_train, y_train, x_test, y_test = GetData(args, timestep)
  model = MakeModel(numFeatures, timestep, signal_count)
  #train model
  model.fit(np.array(x_train), np.array(y_train), 
            batch_size=numFeatures, 
            epochs=5,
            validation_data=[x_test, y_test])
  model.save("./models/"+args.model_name+".h5")
  print("Model saved.")

def Test(args, timestep):
  print("Loading Test Data")
  x_test, y_test = GetData(args)
  print("Loading model")
  model = tf.keras.models.load_model("./models/"+args.model_name+".h5")
  print("Making predictions on test data")
  Predict(model, x_test, y_test)

def main():
  #grab and validate the user arguments
  args = grab_args()

  numFeatures = 128
  timestep = 200
  signal_count = 1001

  if args.testOrTrain == "train":
    Train(args, numFeatures, timestep, signal_count)
  elif args.testOrTrain == "test":
    Test(args, timestep)

if __name__ == "__main__":
  main()