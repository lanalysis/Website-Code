#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
np.object = object    
np.int = int
np.bool = bool    
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  InputLayer
from tensorflow.keras.layers import  Dense

df = pd.read_csv(r'C:\Users\TReAD Lab\Desktop\Roman Machine Learning Models\ML_VETRUN_Data - Sheet1.csv')

df = df.fillna(0)
#print(df)
df1= df.loc[:,'rateExerciseCompletion':'pLowEffortLowRewardPressRate']
df1 = df.drop(["SHAPS_pre","PANAS_pos_pre","PANAS_neg_pre","BDI_pre","AES_pre","AMI_pre","MAPSR_pre","PSS_pre","SHAPS_post","PANAS_pos_post","PANAS_neg_post","BDI_post","AES_post","AMI_post",'MAPSR_post','PSS_post','SHAPS_change','PANAS_pos_change','PANAS_neg_change','BDI_change','AES_change','AMI_change','MAPSR_change','PSS_change','PE_run_days','sleepQuality','drugs','distance','timeRan','pace','effortRun','dEnthusiastic','dCheerful','dRelaxed','dIrritable','dAnxious','dSad','dStressed','dMotivation','dLikelihood','dEffortful','dReward','dTime','dDistance','MA','CA','depressed','stressed','tired','anxious','controllingWorry','worry','rewardRun','effortRun','dRunEnthusiastic','dRunCheerful','dRunRelaxed','dRunIrritable','dRunAnxious','dRunSad','dRunStressed','dRunMotivated','dRunLikelihood','dRunEffortProspect','dRunRewardProspect','dRunTimeProspect','dRunDistanceProspect'], axis=1)
df1 = df.drop(['SUBID','Expected_run_days','Completed_run_days','EMA_submissions'], axis =1)


X = df.loc[:,'calib':'pLowEffortLowRewardPressRate']
#print(X)

y = df.loc[:,'rateExerciseCompletion']
#print(y)


# In[2]:


from keras.layers import Dropout
import keras as keras
from sklearn.model_selection import train_test_split
X = df.loc[:,'calib':'pLowEffortLowRewardPressRate']
X = X.loc[:,['ctRmHighEffortHighReward','ctRmMedEffort','ctRmLowEffortMedReward','LowEffortLowRewardPts','ctRmMedReward','ctRmMedEffortMedReward','HighEffortLowRewardTime']]
             
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 42)

epochs = 100000

model = Sequential()
dropout_layer = Dropout(rate=0.9)
model.add(InputLayer(input_shape=(X_train.shape[1],)))
#add a hidden layer
model.add(Dense(4096, activation = 'sigmoid', kernel_regularizer='l2'))
model.add(Dense(2048, activation = 'sigmoid',kernel_regularizer='l2'))
model.add(Dense(4096, activation = 'sigmoid'))
model.add(dropout_layer)
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='MSE', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=[tf.keras.metrics.MeanSquaredError()])
history = model.fit(X_train, y_train, validation_split=0.15, epochs = epochs, batch_size =16, verbose = 2)
#print(model.evaluate(X_train,y_train))
loss, avg_mse = model.evaluate(X_train,y_train)


# In[21]:


epochdata = pd.DataFrame(history.history)
#epochdata = epochdata.drop(["loss","val_loss","val_mean_squared_error"],axis = 1)
print(epochdata["val_mean_squared_error"].mean())
print(epochdata["mean_squared_error"].mean())


count = 0

for i in range (p):
    if permute_df.iloc[i,0]<0.03814:
        count+=1
p_value = (count+1/(p+1))
print(p_value)
print(count)


# In[9]:


from sklearn.metrics import mean_squared_error

predictions = model.predict(X)
predictions = pd.DataFrame(predictions)
#round predictions to 1 decimal values
predictions[0] = predictions[0].round(1)
y = np.round(y, decimals=1)

#y = y.to_frame()

summation=0
for i in range (19):
    summation+= ((predictions.iloc[i,0]-y.iloc[i,0])**2)
print(summation)
average_error = (summation/19)
print(average_error)

predictionstest = model.predict(X_test)
predictionstest = pd.DataFrame(predictionstest)
predictionstest[0] = predictionstest[0].round(1)
y_test = np.round(y_test, decimals=1)


y_test = y_test.to_frame()


summationtest=0
for i in range (3):
    summationtest+= ((predictionstest.iloc[i,0]-y_test.iloc[i,0])**2)
print(summationtest)
average_errortest = (summationtest/3)
print(average_errortest)


# In[10]:


#p for permutations
p = 10000
permute_df = pd.DataFrame(columns=['permuted_avg_mse'])

for i in range (p):
    df1['rateExerciseCompletion'] = df1['rateExerciseCompletion'].sample(n=19).values
    X = df1.loc[:,['ctRmHighEffortHighReward','ctRmMedEffort','ctRmLowEffortMedReward','LowEffortLowRewardPts','ctRmMedReward','ctRmMedEffortMedReward','HighEffortLowRewardTime']]
    y = df1.loc[:,['rateExerciseCompletion']]
    loss1, avg_mse1 = model.evaluate(X,y, batch_size=16,verbose=1)
    permute_df.loc[i, 'permuted_avg_mse'] = avg_mse1

print(permute_df)
count = 0
for i in range (p):
    if permute_df.iloc[i,0]<avg_mse:
        count+=1
p_value = (count+1/(p+1))
print(p_value)


# In[13]:


for i in range (p):
    if permutetsa_df.iloc[i,0]<0.13:
        count+=1
p_value = (count+1/(p+1))
print(count)


# In[22]:


model.save('regressionrun.keras')

