#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import altair as alt

from pycaret.classification import *
from imblearn.over_sampling import SMOTENC


# # The Data

# In[2]:


data = pd.read_csv("./heart_2020.csv")
data.head(10)


# In[3]:


data.info()


# In[4]:


print("Number of Datapoints:", len(data))
data.HeartDisease.value_counts().plot.bar()


# ### Data Oversampling – SMOTE
# The above chart suggests that the data is severely imbalanced. To overcome this, we will oversample the data using SMOTE.

# In[ ]:


# Oversample with SMOTE
# oversample = SMOTENC(
#     categorical_features=[1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16],
#     sampling_strategy='not majority',
#     k_neighbors=512
# )

# data_labels = np.where(data.HeartDisease.to_numpy() == "Yes", 1, 0)
# features, labels = oversample.fit_resample(data.to_numpy()[:,1:], data_labels)

# print("Number of Datapoints:", len(features))

# features_and_labels = np.hstack((features, labels.reshape(-1, 1)))
# smote_dataset = pd.DataFrame(features_and_labels, columns=[*data.columns[1:], "HeartDisease"])

# smote_dataset   


# # The Model

# In[ ]:


clf1 = setup(data, target='HeartDisease', session_id=123, experiment_name='hhi1')


# ## Compare Baselines

# In[ ]:


best_model = compare_models()


# In[ ]:


et_model = create_model('lightgbm')


# In[ ]:


plot_model(et_model)


# In[ ]:


plot_model(et_model, plot='confusion_matrix')


# In[ ]:


plot_model(et_model, plot='boundary')


# In[ ]:


plot_model(et_model, plot='feature')


# In[ ]:


plot_model(et_model, plot='class_report')


# In[ ]:


# finalize the model
finalModel = finalize_model(et_model)


# In[ ]:


save_model(finalModel, "final_model")
save_config("model_config")

