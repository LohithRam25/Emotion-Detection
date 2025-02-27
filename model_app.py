import pickle
with open("D:\\MACHINE LEARNING\\PROJECTS\\emotion detection\\emotion_model.pkl","rb") as f:
  model=pickle.load(f)
with open("D:\\MACHINE LEARNING\\PROJECTS\\emotion detection\\vectorizer.pkl","rb") as f1:
  vectorizer=pickle.load(f1)
input1=["I am so sad"]
input_features=vectorizer.transform(input1)
prediction=model.predict(input_features)
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


