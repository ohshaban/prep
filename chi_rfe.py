#!/usr/bin/env python
# coding: utf-8

# ### Step 1: Import Functions
# Before we start, we first import functions that we will use from different libraries. 

# ### Step 2: Load data
# 
# The CyTOF data includes has over 47 million cells, 23 proten markers. 
# 
# * The **CyTOF data** contains the single-cell profile of 23 markers for the 2 COVID Conditions - ICU, Ward. See [Source](http://flowrepository.org/id/FR-FCM-Z2KP) for the FCS files. The dimension of the Numpy array is 3 conditions x 47 million cells x 23 markers.

# 
# ### Step 3: Cell Expression  
# 
# To know if there are lowly expressed cells and determine if imputation is required
# 
# Although 15 proteins exhibits approximately 90% cells representation, the cell sparsity in the other 8 proteins might account for irregularities in the model training and prediction. Hence, we will do further preprocessing.

# ### Step 4: Filtering the data
# After loading the data, we're going  to determine the molecule per cell and molecule per gene cutoffs with which to filter the data, in order to remove lowly expressed genes and cells with a small library size.
# 

# ### STEP 5: Normalization and Transformation of Data
# 
# As with any quantitative technology, there is a fundamental need for quality assurance and normalization protocols. The purpose of normalizing(scaling) the data is to make sure that each protein in our counts matrix is counted equally. For instance, in Euclidean distance between two cells, genes that are more highly expressed (i.e. have larger values) will be considered more important.
# 
# - The cofactor parameter adjusts "compression" around zero by dividing the input by this value.
# - Archsin: symmetric make the distributions more symmetric and to map them to a comparable range of expression, which is important for clustering.
# - In CyTOF, the arcsinh transform is popular. 

# ### Step 6 Running MAGIC
# Now the data has been preprocessed, we can run MAGIC with default parameters: knn=5, decay=1, t=3

# ### Step 7 MUltiLayer Perceptron
# We will fit the model on train and test set

# In[2]:


##### Step 1: import functions #####
import pandas as pd
import numpy as np
from numpy.random import seed; seed(111)
import random
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


import os
import pickle
import magic
import scprep
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# In[22]:


with open("magicdata.pkl", "rb") as file:
    data = pickle.load(file)
with open("filtered_nomagicdata.pkl", "rb") as file:
    filtered_nomagicdata = pickle.load(file)
with open("orig_data.pkl", "rb") as file:
    orig_data = pickle.load(file)
filtered_nomagicdata['label'] = data.label
orig_data['label'] = 'healthy'
orig_data.loc[['ward'], 'label'] = 'ward'
orig_data.loc[['icu'], 'label'] = 'icu'


# In[25]:


X = filtered_nomagicdata.drop('label', axis = 1)
X = X.values
cl = {'healthy': 0, 'ward': 1, 'icu': 2}
Y = data["label"].map(cl)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y,random_state=1)


# In[10]:


clf = MLPClassifier(hidden_layer_sizes=(200,), activation='relu', 
                    solver='adam', alpha=0.0001, batch_size='auto', 
                    learning_rate='adaptive', learning_rate_init=0.005, 
                    power_t=0.5, max_iter=200, shuffle=True, random_state=None, 
                    tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
                    nesterovs_momentum=True, early_stopping=False, 
                    validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 
                    epsilon=1e-08, n_iter_no_change=10, max_fun=300).fit(X_train, y_train)
print(clf.score(X_test, y_test))


# ## Average (%) Cell Population across the 3 Disease Conditions

# In[14]:


def cellpopulation(all_data):   
   trys = all_data.groupby('label').mean().T.apply(lambda x: x*100/sum(x), axis=1)
   color = {'healthy': 'blue', 'ward': 'orange', 'icu': 'red'}

   fig, ax = plt.subplots(figsize=(30, 15))
   ax = trys.plot.bar(color=color, 
                  width=0.7, ax = ax)
   for p in ax.patches:
       ax.annotate(format(p.get_height(), '.1f'), 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'left', va = 'center', 
                      size=15,
                     color = 'black',
                      xytext = (-5, 25), 
                   rotation = 90,
                      textcoords = 'offset points')
   ax.set_xticklabels(labels = list(trys.index), rotation=30)
   plt.xlabel('Proteins')
   plt.ylabel('Average (%) Cell Population') 
   plt.legend(trys.columns, loc = 'upper right')
   plt.title('Average (%) Cell Population across the 3 conditions') 

   plt.show()
   None
   
cellpopulation(filtered_nomagicdata)


# The percentage population shows that each proteins have goood representation of each the disease conditions 

# ### Visualizing Relationships
# 
# Note that the change in absolute values of gene expression is not meaningful - the relative difference is all that matters.
# 
# #### Protein-Protein Pairwise MI 

# In[15]:


# Protein-Protein Pairwise MI
def Protein2ProteinMI(data_mc):
    varss = list(data_mc.drop('label', axis = 1).columns) # set independent/dependent vars

    row, mis = [], []
    for var in varss:
        for var2 in varss:
            if var != var2:
                mi = scprep.stats.mutual_information(data[var], data[var2])
                row.append((var, var2))
                mis.append(mi)

    df_mi = pd.DataFrame(mis, index = row, columns = ['MI'])
    df_mi = (df_mi.sort_values(by = 'MI', ascending = False)).iloc[::2, :]
    # df_mi
    df_mi.iloc[0:23].plot.barh(figsize=(20, 8))
    plt.xlabel('MI Score')
    plt.ylabel('Gene Pair')
    plt.title('Protein-Protein Mutual Information')
    None
    
Protein2ProteinMI(data)


# ## Feature Selection
# 
# ### SelectKBest

# In[16]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature extraction
fit = SelectKBest(score_func=chi2, k=6).fit(X, Y)

indices = np.argsort(fit.pvalues_)

feat_importances = (pd.Series(1/np.log(fit.scores_[indices]), index=(data.drop('label', axis = 1).columns)[indices]).sort_values(ascending = True))
#For consistency in the order of feature ranking,
#The inverse of log converts the score to ranking
def plotfe(series, ptitle):
    x=[i for i in range(0,len(series))]
    sns.set(rc={'figure.figsize':(20,12)})
    sns.set_style("whitegrid")
    gcolors = plt.cm.PuBu_r(np.linspace(0, 0.5, len(series)))
    plt.bar(x, series, align="center", color=gcolors, tick_label=(series.index))
    plt.xlabel('Features(protein)')
    plt.ylabel('Ranking')
    plt.title(ptitle)
    plt.savefig(ptitle)
    plt.show()
plotfe(feat_importances, 'SelectKBest')


# ## RFE

# In[17]:


from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

model = DecisionTreeClassifier()
rfe = RFE(model, n_features_to_select=6).fit(X, Y)

indices = np.argsort(rfe.ranking_)
feat_importances = pd.Series(1+np.log(rfe.ranking_[indices]), index=(data.columns)[:-1][indices]).sort_values(ascending = True)
plotfe(feat_importances, 'RFE')


# ### Scanpy Feature Variability Measure

# In[18]:


import scanpy as sc
import scanpy.external as sce
adata = sc.AnnData(data.drop('label', axis = 1))
adata


# In[19]:


adata.obs['label'] = [x for x in data.label]
adata


# In[20]:


# Top expressed genes
sc.pl.highest_expr_genes(adata[:,:-1], n_top=20)

