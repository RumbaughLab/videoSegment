import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import glob
import os
exp=r'C:\Users\Windows\Desktop\Lab\2020-06-06 - grantGavinVid\file'
path = r'C:\Users\Windows\Desktop\Lab\2020-06-06 - grantGavinVid'
vname= '628shockPRO_cut_9809' #'628shock_cut_9605' #628shockPRO_cut_9809
file = glob.glob(path + '/' + vname + '*.h5')[0]
scorer = file.split(os.sep)[-1].split('.h5')[0].split(vname)[-1]
Dataframe = pd.read_hdf(file, "df_with_missing")
bodyparts2plot = ['snoutL', 'snoutR', 'snoutTip']
custdpi=100

for kk in range(0,len(Dataframe)):
    figure = plt.figure()
    fac=15
    plt.plot([340+fac,410-fac], [600-fac,545+fac], '.',alpha=0)
    for i in range(1,10):
        for bpindex, bp in enumerate(bodyparts2plot):
            plt.plot(Dataframe[scorer][bp]['x'].values[i], Dataframe[scorer][bp]['y'].values[i], 'o',
                     color='orange', alpha=i/10, markersize=10)

        yval = [Dataframe[scorer]['snoutL']['y'].values[i],
                Dataframe[scorer]['snoutR']['y'].values[i],
                Dataframe[scorer]['snoutTip']['y'].values[i],
                Dataframe[scorer]['snoutL']['y'].values[i]]
        xval = [Dataframe[scorer]['snoutL']['x'].values[i],
                Dataframe[scorer]['snoutR']['x'].values[i],
                Dataframe[scorer]['snoutTip']['x'].values[i],
                Dataframe[scorer]['snoutL']['x'].values[i]]

        plt.plot(xval, yval, color='orange', alpha=i/10)

    plt.axis('off')
    plt.gca().invert_yaxis()

    figure.savefig(exp+'/test' + f"{kk:08d}" + '.png', pad_inches=0,
                   dpi=custdpi, transparent=True)  # xdimIm and ydimIm can be modified here for croping pupuse
    plt.close('all')

ffmpeg -r 1/5 -i img%08d.png -c:v libx264 -vf fps=30 out.mp4