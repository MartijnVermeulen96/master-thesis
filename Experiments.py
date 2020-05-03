#!/usr/bin/env python
# coding: utf-8

# In[1]:


import errorAPI
from errorAPI import Dataset

datasets = Dataset.list_datasets()
dataset_filter_out = ["company", "tax"]
datasets = [x for x in datasets if x not in dataset_filter_out]
datasets


# ## Experiment run
# This will run all the given example configurations on the datasets specified.
# When a single experiment is done, it will be uploaded in a specified SQL schema in table 'results'

# In[3]:


confirm_new = input("Create new experiments / load otherwise? (y/N):")

if confirm_new in ['y', 'Y']:
    sql_string = 'postgresql://postgres:postgres@localhost:5432/error_detection'
    experiment = errorAPI.Experiment.create_example_configs(sql_string, datasets)
else:
    print("Loading the experiments state")
    experiment = errorAPI.Experiment.load_experiment_state()


# In[ ]:





# In[4]:


print("Experiments in queue:", len(experiment.experiments_q))
print("Experiments done:", len(experiment.experiments_done))

# In[7]:


verbose_in = input("Verbose print expirements? (y/N):")
if verbose_in in ['y', 'Y']:
    experiment.no_print = False
else:
    experiment.no_print = True

try:
    timeout = int(input("Timeout in seconds (1800): "))
except:
    timeout = 1800

experiment.timeout = timeout
# In[ ]:


confirm_in = input("Run it all? (y/N):")

if confirm_in in ['y', 'Y']:
    experiment.run()


# In[ ]:





# In[ ]:





# In[ ]:




