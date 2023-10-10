import numpy as np
import pickle 

#loading the saved model
loaded_model=pickle.load(open("D:/Skills/project/ML project/Diabetes Prediction/trained_model.sav", 'rb'))


input_data=(9,171,110,24,240,45.4,0.721,54)


#changig the input_data to numpy array
input_data_as_numpy_array =np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)


prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print('The person is non Diabetic')
else:
  print('The person is Diabetic')