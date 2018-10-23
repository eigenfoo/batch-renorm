import glob
import pandas as pd

for norm_type in ['norm', 'renorm']:
    filenames = glob.glob('./val_accs_{}_*.csv'.format(norm_type))
    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filename, index_col=0))
    mean_df = pd.concat(dfs, axis=1).mean(axis=1)
    mean_df.to_csv('val_accs_{}_mean.csv'.format(norm_type))
