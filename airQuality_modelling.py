#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pprint import PrettyPrinter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import mean_absolute_error
from pymongo import MongoClient
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model  import AutoRegResultsWrapper


# In[ ]:


#Requirements
pp = PrettyPrinter(indent=2)
host = "192.21.158.2"
client = MongoClient(host, port=27017)
#Checking the content of the database
for data_base in (client.list_databases()):
    print(data_base["name"])
#navigating to the air-quality database
db = client["air-quality"]
#navigating to the dar-es-salam collection
dar = db["dar-es-salaam"]


# In[ ]:


#counting the site with the largest number of readings
result = result = dar.aggregate(
    [
        {"$group": {"_id": "$metadata.site", "count":{"$count":{}}}}
    ]
)
readings_per_site = list(result)
readings_per_site


# In[3]:


#Wrangle function that preprocesses the data
def wrangle(collection, resample_rule="1H"):
    result = collection.find(
    {"metadata.site":11, "metadata.measurement":"P2"},
    projection ={"P2":1, "timestamp":1, "_id":0}
    )
    y = pd.DataFrame(list(result)).set_index("timestamp")
    #Remove Outliers
    y =y[y["P2"] <= 100]
    #localize time and convert to Africa/Nairobi
    y.index = y.index.tz_localize("utc").tz_convert("Africa/Dar_es_Salaam")
    #resample to one hour and forward fill null values
    y = y["P2"].resample(resample_rule).mean().fillna(method="ffill")

    return y


# In[ ]:


#storing the preprocessed data
y = wrangle(dar)
print(type(y))
y


# In[ ]:


#Time series plot of y
fig, ax = plt.subplots(figsize=(15, 6))
y.plot(ax=ax, xlabel= "Date", ylabel="PM2.5 Level", title="Dar es Salaam PM2.5 Levels")


# In[ ]:


#Plot of the rolling average of y
fig, ax = plt.subplots(figsize=(15, 6))
y.rolling(168).mean().plot(xlabel="Date", ylabel="PM2.5 Level", title="Dar es Salaam PM2.5 Levels, 7-Day Rolling Average", ax=ax)


# In[ ]:


#Autocorrelation function ACF plot
fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y, ax=ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Correlation Coefficient")
plt.title("Dar es Salaam PM2.5 Readings, ACF")


# In[ ]:


#Partial Aurocorrelation function PACF plot
fig, ax = plt.subplots(figsize=(15, 6))
plot_pacf(y, ax=ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Correlation Coefficient")
plt.title("Dar es Salaam PM2.5 Readings, PACF")


# In[ ]:


#Model building section
#Splitting
cutoff_test = int(len(y)*0.9)
y_train = y[:cutoff_test]
y_test = y[cutoff_test:]
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[ ]:


#baseline
y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean]*len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean P2 Reading:", y_train_mean)
print("Baseline MAE:", mae_baseline)


# In[ ]:


#Iterate

# Create empty list to hold mean absolute error scores
maes = []
p_params = range(1, 31)
# Iterate through all values of p in `p_params`
for p in p_params:
    # Build model
    model = AutoReg(y_train, lags=p).fit()

    # Make predictions on training data, dropping null values caused by lag
    y_pred = model.predict().dropna()

    # Calculate mean absolute error for training data vs predictions
    mae = mean_absolute_error(y_train.iloc[p:], y_pred)

    # Append `mae` to list `maes`
    maes.append(mae)

# Put list `maes` into Series with index `p_params`
mae_series = pd.Series(maes, name="mae", index=p_params)

# Inspect head of Series
mae_series.head()


# In[ ]:


#using value of p in p_params in which model performance is the best (in terms of mae)
best_p = 26
best_model = AutoReg(y_train, lags=best_p).fit()


# In[ ]:


#Getting the residuals
y_train_resid = best_model.resid
y_train_resid.name = "residuals"
y_train_resid


# In[ ]:


# Plot histogram of residuals
fig, ax = plt.subplots()
y_train_resid.hist(ax=ax)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Best Model, Training Residuals")


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y_train_resid, ax=ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Correlation Coefficient")
plt.title("Dar es Salaam, Training Residuals ACF")


# In[ ]:


#ACF plot of residuals 
fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y_train_resid, ax=ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Correlation Coefficient")
plt.title("Dar es Salaam, Training Residuals ACF")


# In[ ]:


#Using walk forward validation to train model
y_pred_wfv = pd.Series()
history = y_train.copy()
for i in range(len(y_test)):
    model = AutoReg(history, lags=26).fit()
    next_pred = model.forecast()

    y_pred_wfv = y_pred_wfv.append(next_pred)
    history = history.append(y_test[next_pred.index])


y_pred_wfv.name = "prediction"
y_pred_wfv.index.name = "timestamp"
y_pred_wfv.head()


# In[ ]:


#Evaluating the model
test_mae = mean_absolute_error(y_test, y_pred_wfv)
print("Test MAE (walk forward validation):", round(test_mae, 2))


# In[ ]:


#communicating result
df_pred_test = {"y_test": y_test, "y_pred_wfv": y_pred_wfv}
fig = px.line(df_pred_test)
fig.update_layout(
    title="Dar es Salaam, WFV Predictions",
    xaxis_title="Date",
    yaxis_title="PM2.5 Level",
)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




