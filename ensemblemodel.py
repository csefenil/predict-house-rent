import pandas as pd
import numpy as np
from math import sqrt
import warnings
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

import pickle

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score

from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

ds=pd.read_csv("MagicBricks.csv")
ds.columns=['Area','BHK','Bathroom','Furnishing','Locality','Parking','Transaction','Type','Per_Sqft','Total_Sqft','Rent']
ds.info(), ds

def set_furnishing(val):
    if val=='Unfurnished':
        return 0
    elif val == 'Semi-Furnished':
        return 1
    else:
        return 2
        
def set_type(val):
    if val=='Builder_Floor':
        return 0
    else:
        return 1


def set_transaction(val):
    if val=='Resale':
        return 0
    else:
        return 1
    
ds['Furnishing']=ds['Furnishing'].apply(set_furnishing)
ds['Type']=ds['Type'].apply(set_type)
ds['Transaction']=ds['Transaction'].apply(set_transaction)

def set_locality(val):
    if val=='Alaknanda':
        return 1
    elif val=='Budh Vihar':
        return 2
    elif val=='Chhattarpur':
        return 3
    elif val=='Chittaranjan Park':
        return 4
    elif val=='Commonwealth Games Village':
        return 5
    elif val=='Dilshad Garden':
        return 6
    elif val=='Dwarka Sector':
        return 7
    elif val=='Geeta Colony':
        return 8
    elif val=='Greater Kailash':
        return 9
    elif val=='Hauz Khas':
        return 10
    elif val=='Janakpuri':
        return 11
    elif val=='Kalkaji':
        return 12
    elif val=='Karol Bagh':
        return 13
    elif val=='Kilokri':
        return 14
    elif val=='Kirti Nagar':
        return 15
    elif val=='Khairatabad':
        return 16
    elif val=='Lajpat Nagar':
        return 17
    elif val=='Laxmi Nagar':
        return 18
    elif val=='Madangir':
        return 19
    elif val=='Malviya Nagar':
        return 20
    elif val=='Mehrauli':
        return 21
    elif val=='Najafgarh':
        return 22
    elif val=='Narela':
        return 23
    elif val=='New Friends Colony':
        return 24
    elif val=='Okhla':
        return 25
    elif val=='Paschim Vihar':
        return 26
    elif val=='Patel Nagar':
        return 27
    elif val=='Punjabi Bagh':
        return 28
    elif val=='Rohini Sector':
        return 29
    elif val=='Safdarjung':
        return 30
    elif val=='Saket':
        return 31
    elif val=='Sarita Vihar':
        return 32
    elif val=='Shahdara':
        return 33
    elif val=='Sheikh Sarai':
        return 34
    elif val=='Sultanpur':
        return 35
    elif val=='Uttam Nagar':
        return 36
    elif val=='Vasant Vihar':
        return 37
    elif val=='Vasundhara':
        return 38

ds['Locality'] = ds['Locality'].apply(set_locality)

ds.head()



df=ds[['Area','BHK','Bathroom','Furnishing','Locality','Parking','Transaction','Type','Per_Sqft','Total_Sqft','Rent']]
df=df.fillna(0)
df.Area.astype(int)
df.Bathroom.astype(int)
df.Parking.astype(int)
df.Per_Sqft.astype(int)
df.Total_Sqft.astype(int)

dt=df

x=dt.iloc[:,0:10].values
y=dt.iloc[:,10].values
x.shape,y.shape,dt.head()

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.30,random_state=1)
xtrain.shape,xtest.shape,ytrain.shape,ytest.shape

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model = Sequential()
model._estimator_type = "regressor"
model.add(Dense(51,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(12,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

model.fit(x=xtrain,y=ytrain,
          validation_data=(xtest,ytest),
          epochs=1000,callbacks=[early_stop])

predictions = model.predict(xtest)
print("RMSE for NN: ",np.sqrt(mean_squared_error(ytest,predictions)))
print("Explained Variance: ",explained_variance_score(ytest,predictions))

dr=DecisionTreeRegressor()
rf=RandomForestRegressor()
drmodel = dr.fit(xtrain,ytrain)
rfmodel = rf.fit(xtrain,ytrain)

dr_pred=drmodel.predict(xtest)
rf_pred=rfmodel.predict(xtest)

voting = VotingRegressor([('DR', dr), ('RF', rf)])
voting_predict=voting.fit(xtrain, ytrain).predict(xtest)

nn_predict = []
for i in predictions:
  a = [i]
  nn_predict.extend(i)

model_predict = (voting_predict + nn_predict)/2

voting_mse = mean_squared_error(ytest,voting_predict)
print(f"RMSE for ensemble voting regressor = ",sqrt(voting_mse))

dr_score = r2_score(ytest,dr_pred)
rf_score = r2_score(ytest,rf_pred)

print("dr score=",dr_score,"\nrf score=",rf_score)

dr_mse = mean_squared_error(ytest,dr_pred)
rf_mse = mean_squared_error(ytest,rf_pred)

print(f"rmse for decision tree regressor: ",sqrt(dr_mse),"\nrmse for random forest regressor: ",sqrt(rf_mse))

voting_score = r2_score(ytest,voting_predict)

print("voting score=",voting_score)

model_score = r2_score(ytest, model_predict)
print(model_score)

pickle.dump(drmodel, open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
print(model.predict([[900,2,2,1,1,1,1,1,190,900]]))

# from tensorflow import lite
# converter = lite.TFLiteConverter.from_keras_model(model)
#
# tfmodel = converter.convert()
#
# open('hrp.tflite','wb').write(tfmodel)
#
# pickle.dump(model_predict,open('model.pkl','wb'))
