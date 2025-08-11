import joblib 
import numpy as np 
model = joblib.load("personsai_model.pkl")
y = model.predict(np.array([[25, 1, 2, 170]]))  
print(y)