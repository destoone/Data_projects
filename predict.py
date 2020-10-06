import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sphist= pd.read_csv("sphist.csv")
sphist["Date"]= pd.to_datetime(sphist["Date"])

## let's look at if converting be matching
#print(sphist["Date"] > datetime(year=2015, month=4, day=1))

sphist.sort_values(by="Date",ascending=True,inplace=True)
#print(sphist["Date"].head())

### --- Let's create 3 indicators and generate column for each one --- ###

data_mean_5day = sphist.Close.rolling(5).mean().shift(1)
data_mean_30day = sphist.Close.rolling(30).mean().shift(1)
data_mean_365day = sphist.Close.rolling(365).mean().shift(1)
data_std_5day = sphist.Close.rolling(5).std().shift(1)
data_std_365day = sphist.Close.rolling(365).std().shift(1)

sphist["day_5"]= data_mean_5day
sphist["day_30"]= data_mean_30day
sphist["day_365"]= data_mean_365day
sphist["day_s_5"]= data_std_5day
sphist["day_s_365"]= data_std_365day
#print(sphist.head(10))

sphist= sphist[sphist["Date"]> datetime(year=1951,month=1,day=3)]
sphist= sphist.dropna(axis=0)

# Let's split our data in train and test
val_index= len(sphist.loc[sphist["Date"]< datetime(year=2013,month=1,day=1),"Date"])

train= sphist[:val_index]
test= sphist[val_index:]

#Now let's define an error metric
training= ["day_5","day_30","day_365","day_s_5","day_s_365"]
target= "Close"
lr= LinearRegression()
lr.fit(train[training],train[target])
predictions= lr.predict(test[training])
mae= np.mean((test[target]-predictions).abs())
rmse= mean_squared_error(predictions,test[target])**1/2
print(mae,rmse)


