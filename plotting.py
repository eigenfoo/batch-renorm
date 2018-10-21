import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('val_accs_norm.csv', index_col=0)
dff = pd.read_csv('val_accs_renorm.csv', index_col=0)

fig, ax = plt.subplots(figsize=[12, 8])
ax.plot(df, 'b--', label='Batch Normalization')
ax.plot(dff, 'r', label='Batch Renormalization')
ax.set_xlabel('Training steps')
ax.set_ylabel('Accuracy')
ax.set_xticklabels(['{:}k'.format(int(x/1000)) for x in ax.get_xticks()]);
ax.set_yticklabels(['{:.0%}'.format(y) for y in ax.get_yticks()])
ax.legend()
ax.grid(True)

fig.savefig('figure.png')
