# Loading Libraries we need to use

import numpy as np
import pandas as pd

# For plotting
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

#For standardising the dat
from sklearn.preprocessing import StandardScaler

#PCA!!
from sklearn.decomposition import PCA

# Loading the data from Kannada-MNIST folder

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv('../manzitlo/Kannada-MNIST/train.csv')
test = pd.read_csv('../manzitlo/Kannada-MNIST/test.csv')
train.head()

## Setting the label and the feature columns
y = train.loc[:,'label'].values
x = train.loc[:,'pixel0':].values

## Standardizing the data
standardized_data = StandardScaler().fit_transform(x)
print(standardized_data.shape)

## Importing PCA

pca = PCA(n_components=2) # project from 784 to 2 dimensions
principalComponents = pca.fit_transform(x)
principal_df = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principal_df.shape

# Explaining the Variance ratio

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# Plot the first two principal components of each point to learn about the data:

plt.scatter(principalComponents[:, 0], principalComponents[:, 1], s= 5, c=y, cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('Visualizing Kannada MNIST by using PCA', fontsize=20);
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

