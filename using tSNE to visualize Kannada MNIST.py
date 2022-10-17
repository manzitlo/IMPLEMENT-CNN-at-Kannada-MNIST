# Loading the libraries that we need to use
from time import time

import numpy as np
import pandas as pd


# For plotting
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import plotly.graph_objects as go

%matplotlib inline
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

#For standardising the dat
from sklearn.preprocessing import StandardScaler

#TSNE !!
from sklearn.manifold import TSNE

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Reading the data
train=  pd.read_csv('../manzitlo/Kannada-MNIST/train.csv')
test=  pd.read_csv('../manzitlo/Kannada-MNIST/test.csv')

train.head()

# Setting the label and the feature columns
y = train.loc[:,'label'].values
x = train.loc[:,'pixel0':].values

print(x.shape)
print(y)

# Plotting the original train data
def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(28, 28),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))
plot_digits(x)

## Standardizing the data
standardized_data = StandardScaler().fit_transform(x)
print(standardized_data.shape)

# t-SNE is consumes a lot of memory so we shall use only a subset of our dataset. 
x_subset = x[0:10000]
y_subset = y[0:10000]

print(np.unique(y_subset))

# Applyting t-SNE on the data
%time
tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=300).fit_transform(x_subset)

# Visualizing the t-SNE
plt.scatter(tsne[:, 0], tsne[:, 1], s= 5, c=y_subset, cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('Visualizing Kannada MNIST by using t-SNE', fontsize=20);

