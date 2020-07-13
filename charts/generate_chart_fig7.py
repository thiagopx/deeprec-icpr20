import sys
import json
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
font_manager._rebuild()

from matplotlib import rc
rc('text', usetex=True)
# rc('text.latex', preamble=r'\usepackage{amsmath}')
rc('font',**{'family':'serif','serif':['Nimbus Roman No9 L']})

import seaborn as sns
sns.set(context='paper', style='darkgrid', font_scale=7)
# sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})

# datasets = ['D1', 'D2', 'D3']
results_deep = json.load(open('results/deep.json', 'r'))
results_deep_ma = json.load(open('results/deep-ma.json', 'r'))
records = []
for entry in results_deep + results_deep_ma:
    method = entry[0]
    _, dataset, _, doc_id = entry[1].split('/')
    acc = entry[2]
    records.append([method, dataset, doc_id, 100* acc])

df = pd.DataFrame.from_records(records, columns=('method', 'dataset', 'doc', 'accuracy'))
print(df)

fp = sns.catplot(
    x='dataset', y='accuracy', hue='method', hue_order=['deep-ma', 'deep'],
    height=17, aspect=2., kind='box', width=0.5, linewidth=8,
    data=df, legend_out=False
)
# fp = fp.map(sns.lineplot, 'k', 'accuracy', marker='s', ci=95, markersize=50)
fp.add_legend(title='\\textbf{method}', prop={'size': 130}, labelspacing=0.1)

yticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
fp.set(yticks=yticks, ylim=(0, 101))
fp.ax.set_xlabel('dataset', fontsize=130)
fp.ax.set_ylabel('accuracy (\%)', fontsize=130)
fp.set_xticklabels(
    ['\\textsc{S-Marques}', '\\textsc{S-Isri-OCR}', '\\textsc{S-Cdip}', '\\textsc{Images}'],
     fontdict={'fontsize': 115}, rotation=20
)
fp.set_yticklabels(yticks, fontdict={'fontsize': 130})

map_leg_label = {'deep': 'Deep', 'deep-ma': '\\emph{Deep-MA}'}
leg = fp.ax.get_legend()
plt.setp(leg.get_title(), fontsize=130)  # legend title size
for text in leg.get_texts():
    text.set_text(map_leg_label[text.get_text()])

bb = leg.get_bbox_to_anchor().inverse_transformed(fp.ax.transAxes)
dx = 0.525
dy = -0.15
bb.y0 += dy
bb.y1 += dy
bb.x0 += dx
bb.x1 += dx
leg.set_bbox_to_anchor(bb, transform=fp.ax.transAxes)

path = 'charts'
if len(sys.argv) > 1:
    path = sys.argv[1]
plt.savefig('{}/chart_fig7.pdf'.format(path), bbox_inches='tight')