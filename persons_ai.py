import pandas as pd 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense  
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
import joblib
mydata=pd.read_csv('persons.csv') 
le= LabelEncoder()

mydata["Gender_encoded"] = le.fit_transform(mydata[["Gender"]])
mydata["BodyType_encoded"] = le.fit_transform(mydata[["BodyType"]])
x = mydata[["Age","Gender_encoded","BodyType_encoded","Height"]]
y = mydata["Weight"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3) 

model = Sequential()

model.add(Dense(10, activation="relu",input_shape=(4,)))
model.add(Dense(10, activation="relu")) 
model.add(Dense(10, activation="relu"))
model.add(Dense(1))  
model.compile(optimizer="adam", loss="mse")
model.fit(x_train, y_train, epochs=100) 
joblib.dump(model, "personsai_model.pkl")

print(mydata) 

