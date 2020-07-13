import json
import pandas as pd
from scipy.stats import ttest_rel


results_deep = json.load(open('results/deep.json', 'r'))
results_deep_ma = json.load(open('results/deep-ma.json', 'r'))
records = []
for entry in results_deep + results_deep_ma:
    method = entry[0]
    _, dataset, _, doc_id = entry[1].split('/')
    acc = entry[2]
    records.append([method, dataset, doc_id, 100* acc])

df = pd.DataFrame.from_records(records, columns=('method', 'dataset', 'doc', 'accuracy'))

for dataset in ['S-MARQUES', 'S-ISRI-OCR', 'S-CDIP', 'images']:
    print('dataset={}'.format(dataset), end=' :: ')
    data1 = df[(df['method'] == 'deep') & (df['dataset'] == dataset)]['accuracy']
    data2 = df[(df['method'] == 'deep-ma') & (df['dataset'] == dataset)]['accuracy']
    stat, p = ttest_rel(data1, data2)
    print('stat={:.6f}, p={:.6f}, deep={:.2f} +- {:.2f} deep-ma={:.2f} +- {:.2f}'.format(
        stat, p, data1.mean(), data1.std(), data2.mean(), data2.std()
    ), end=': ')
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')