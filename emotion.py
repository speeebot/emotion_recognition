from helpers import *
np.set_printoptions(threshold=sys.maxsize)

#-----------------------Class types-----------------------------
#1: Happy, 2: Surprised, 3: Sad, 4: Startled/Surprised, 5: Skeptical
#6: Embarrassed, 7: Scared/Nervous, 8: Physical Pain, 9: Angry, 10: Disgusted

def MakeModel(num_features, timestep):
  #create sequential model
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.LSTM(num_features, input_shape=(1, 1000), activation='relu', return_sequences=True))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.LSTM(int(num_features/2), activation='sigmoid'))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
  model.summary()
  return model

def Predict(model, x_test, y_test):
  #predict and format output to use with sklearn
  predict = model.predict(x_test)
  predict = np.argmax(predict, axis=1)
  #macro precision, recall, and F1 score
  precision_macro = precision_score(y_test, predict, average='macro')
  recall_macro = recall_score(y_test, predict, average='macro')
  f1_macro = f1_score(y_test, predict, average='macro')
  #micro precision, recall, and F1 score
  precision_micro = precision_score(y_test, predict, average='micro')
  recall_micro = recall_score(y_test, predict, average='micro')
  f1_micro = f1_score(y_test, predict, average='micro')
  #confusion matrix
  confMat = confusion_matrix(y_test, predict)

  print("Macro precision: ", precision_macro)
  print("Micro precision: ", precision_micro)
  print("Macro recall: ", recall_macro)
  print("Micro recall: ", recall_micro)
  print("Macro F1 score: ", f1_macro)
  print("Micro F1 score: ", f1_micro)
  print(confMat)

def Train(args, num_features, timestep, num_epochs):
  print("Loading Training Data")
  #get training data
  x_train, y_train, x_valid, y_valid = GetData(args, timestep)

  #make the model
  model = MakeModel(num_features, timestep)
  #train model
  history = model.fit(x_train, y_train, 
            batch_size=num_features, 
            epochs=num_epochs,
            validation_data=(x_valid, y_valid),
            shuffle=True)
  #save model
  model.save("./models/"+args.model_name+".h5")
  print("Model saved.")
  
  return history

def Test(args, timestep):
  print("Loading Test Data")
  x_test, y_test = GetData(args, timestep)
  print("Loading model")
  model = tf.keras.models.load_model("./models/"+args.model_name+".h5")
  print("Making predictions on test data")
  Predict(model, x_test, y_test)

def main():
  #grab and validate the user arguments
  args = grab_args()

  num_epochs = 50
  num_features = 64
  timestep = 1000

  if args.testOrTrain == "train":
    history = Train(args, num_features, timestep, num_epochs)
    plot_results(history)
  elif args.testOrTrain == "test":
    Test(args, timestep)

if __name__ == "__main__":
  main()
