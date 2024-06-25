#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generating loss vs epoch graphs

input: embeddings_final/output/transE_holE_hyperparam_result.csv OR 
        embeddings_final/output/transH_hyperparam_result.csv
output: embeddings_final/output/transE_holE_hyperparam_result.pdf OR 
        embeddings_final/output/transH_hyperparam_result.pdf

Note: please change the paths to support your directory structure
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
from pylab import *
import numpy as np
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages

# %matplotlib inline

# Load the "transH_hyperparam_result.csv" or "transE_holE_hyperparam_result.csv"
data = pd.read_csv('embeddings_final/output/transH_hyperparam_result.csv')
# data = pd.read_csv('embeddings_final/output/transE_holE_hyperparam_result.csv')

# extract the losses_lists and epochs
data['losses_list'] = data['losses_list'].apply(lambda x:json.loads(x))
epochs = data['epochs']


fig = plt.figure()

# create a pdf file which will store the graps for each of the model present in the earlier loaded csv file
with PdfPages('embeddings_final/output/transH_hyperparam_result.pdf') as pdf:
    
    # plot the loss vs epoch graph for each of the model and save it to the pdf file
    for i in range(len(data)):
        modelname = data['model_name'][i]
        epoch = data['epochs'][i]
        k = data['k'][i]
        lr = data['lr'][i]
        losses = data['losses_list'][i]
        
        name = str(modelname)+"_"+str(epoch)+"_"+str(k)+"_"+str(lr)
        x = list(range(1,epoch+1))
        y = losses    
        plt.title(name, fontsize='small')
        plt.xlabel("Epochs", fontsize='small')
        plt.ylabel("Loss", fontsize='small')
        plt.plot(x, y)  # Plot the chart
        
        pdf.savefig()
        plt.close()
    

