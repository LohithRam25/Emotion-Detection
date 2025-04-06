import pickle
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
with open("D:\\MACHINE LEARNING\\PROJECTS\\emotion detection\\emotion_model.pkl","rb") as f:
  model=pickle.load(f)
with open("D:\\MACHINE LEARNING\\PROJECTS\\emotion detection\\vectorizer.pkl","rb") as f1:
  vectorizer=pickle.load(f1)
from tensorflow.keras.models import load_model

neural_model=load_model("D:\\MACHINE LEARNING\\PROJECTS\\emotion detection\\neural_model.keras")
neural_model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
input1=["please dont kill me"]
input_features=vectorizer.transform(input1)
prediction=neural_model.predict(input_features)
prediction=np.argmax(prediction)
if prediction==0:
  print("Anger")
elif(prediction==1):
  print("Disgust")
elif(prediction==2):
  print("Fear")
elif(prediction==3):
  print("Joy")
elif(prediction==4):
  print("Neutral")
elif(prediction==5):
  print("Sadness")
elif(prediction==6):
  print("shame")
elif(prediction==1):
  print("Surprise")


