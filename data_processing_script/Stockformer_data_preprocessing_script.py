import pandas as pd
import numpy as np
import networkx as nx
import sys
import os

# Relative directory path
directory = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/data/Stock_CN_2021-06-04_2024-01-30'
if not os.path.exists(directory):
    os.makedirs(directory)
    print("Directory created:", directory)

# Reading the data ########
df = pd.read_csv(f'{directory}/label.csv', index_col=0)
df.index = pd.to_datetime(df.index)
# 将nan替换为0
df.fillna(0, inplace=True)
print('Data read successfully.')

# Convert data to NumPy array and save as npz
data = df.values
np.savez(f'{directory}/flow.npz', result=data)
print('Flow array saved as npz successfully.')

# 将收益率转换为涨跌分类：正收益为1，否则为0
trend_indicator = (data > 0).astype(int)
np.savez(f'{directory}/trend_indicator.npz', result=trend_indicator)
print('Trend indicator saved as npz successfully.')

# Check for columns with zero variance and replace them
epsilon = 1e-10  # A small constant
std_devs = np.std(df, axis=0)
zero_variance_mask = std_devs < epsilon
df.loc[:, zero_variance_mask] = epsilon  # Replace zero variance columns with a small constant

# Calculate and save correlation matrix
corr_matrix = np.corrcoef(df, rowvar=False)
np.save(f'{directory}/corr_adj.npy', corr_matrix)
print('Correlation matrix saved successfully.')

# Generate and save edge list
edge_list = []
for i in range(corr_matrix.shape[0]):
    for j in range(i+1, corr_matrix.shape[1]):
        weight = corr_matrix[i, j]
        edge_list.append((i, j, weight))
with open(f'{directory}/data.edgelist', 'w') as f:
    for edge in edge_list:
        f.write('{} {} {}\n'.format(edge[0], edge[1], edge[2]))
print('Edge list saved successfully.')

# Adjusted for relative paths for custom libraries
sys.path.append('/root/autodl-tmp/Stockformer/Stockformer_run/GraphEmbedding')
from ge.classify import read_node_label, Classifier
from ge import Struc2Vec

# Read the graph, train the model, and save embeddings
G = nx.read_edgelist(f'{directory}/data.edgelist', create_using=nx.DiGraph(), nodetype=None, data=[('weight', float)])
model = Struc2Vec(G, 10, 80, workers=4, verbose=40)
model.train(embed_size=128)
embeddings = model.get_embeddings()

# Convert embeddings to numpy array and save
embedding_array = np.array(list(embeddings.values()))
np.save(f'{directory}/128_corr_struc2vec_adjgat.npy', embedding_array)
print('Embedding array saved successfully.')
