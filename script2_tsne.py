import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

# Load features. Assuming features are saved as a numpy array in 'con_features.npy'
# Alternatively, you could load from a mat file if needed.
con_features = np.load('con_features.npy')

# Load data from Excel. Adjust the file name and column names as needed.
excel_file = 'NewMasterSheet.xlsx'
data = pd.read_excel(excel_file)

# Assuming the Excel file has columns 'Context' and 'Site'
ContextCatb = data['Context'].values
Site = data['Site'].values

# Define perplexity values to use
perplexities = [20, 25, 30]

def plot_tsne(features, labels, perplexity, title_suffix=''):
    tsne = TSNE(perplexity=perplexity, random_state=4)
    tsne_results = tsne.fit_transform(features)
    plt.figure()
    # create a scatter plot
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        idx = labels == lab
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=lab, alpha=0.6, s=20)
    plt.title(f't-SNE (Perplexity={perplexity}) {title_suffix}')
    plt.legend()
    plt.show()

# Plot t-SNE for each selected perplexity for Context categories
for perp in perplexities:
    plot_tsne(con_features, ContextCatb, perp, title_suffix='- Context')

# Enhanced plotting with different markers based on a list of positive contexts
contextlabels = np.array(['AfterNursing','Barren','BeforeNursing','Castration','Crushing','Enriched',
                            'Fighting','Handling','Huddling','Isolation','MissedNursing','NegativeCondtioning',
                            'NovelObject','PositiveConditioning','Restrain','Reunion','Run','Surprise','Waiting'])
# Define which contexts are positive
poscons = np.array(['AfterNursing','BeforeNursing','Enriched','Huddling','PositiveConditioning','Reunion','Run'])

# Use one perplexity value (e.g., 25) for enhanced plotting
tsne = TSNE(perplexity=25, random_state=4)
con_tsne = tsne.fit_transform(con_features)

# For color mapping, we'll use a list of distinct colors
colors = cm.get_cmap('tab20', len(contextlabels))

plt.figure()
for j, ctx in enumerate(contextlabels):
    idx = (ContextCatb == ctx)
    if np.sum(idx) == 0:
        continue
    marker = 'o' if ctx in poscons else '^'
    plt.scatter(con_tsne[idx, 0], con_tsne[idx, 1], s=20, alpha=0.6, marker=marker, color=colors(j), label=ctx)
plt.title('t-SNE for Context Labels with Custom Markers')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axis([-85,85,-85,85])
plt.show()

# Additional plot for Site labels if needed
plt.figure()
unique_sites = np.unique(Site)
for site in unique_sites:
    idx = (Site == site)
    plt.scatter(con_tsne[idx, 0], con_tsne[idx, 1], s=20, alpha=0.6, label=site)
plt.title('t-SNE for Site Labels')
plt.legend()
plt.axis([-85,85,-85,85])
plt.show()