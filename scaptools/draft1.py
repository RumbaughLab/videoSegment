
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import glob
import os

def pupilDiam(vname, path, pupilPoint):
    """function returns the average distance in pixels from one frame to the next
    Parameters:
        vname (str): name of the video file to be worked on
        path (str): path where the h5 files are located with the video
        bodyparts (list): list of bodyparts of interest which are related eg. ['snoutL', 'snoutR', 'snoutTip']

    Return:
    	mainTemp (average distance difference in pixels)
    	save the meanFrame timeseries to numpy array

    Usage example:
        # distanceMoved('628shockPRO_cut_9809', 'C:/Users/Windows/Desktop/Lab/2020-06-06 - grantGavinVid', ['snoutL', 'snoutR', 'snoutTip'])
    """
    file = glob.glob(path+'/'+vname+'*.h5')[0]
    scorer = file.split(os.sep)[-1].split('.h5')[0].split(vname)[-1]
    df = pd.read_hdf(file, "df_with_missing")
    df = df[scorer]

    # drop the multi index and change the column names
    df.columns = [''.join(col) for col in df.columns]
    # reset row index
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'frame'}, inplace=True)
    if pupilPoint == 6:
        df = df.assign(diamHpupil=((df.pupil2x - df.pupil1x) ** 2 + (df.pupil2y - df.pupil1y) ** 2) ** 0.5,
                       diamVpupil=((df.pupil4x - df.pupil3x) ** 2 + (df.pupil4y - df.pupil3y) ** 2) ** 0.5,
                       diamDpupil=((df.pupil6x - df.pupil5x) ** 2 + (df.pupil6y - df.pupil5y) ** 2) ** 0.5,
                       eyeH=((df.eye1x - df.eye2x) ** 2 + (df.eye1y - df.eye2y) ** 2) ** 0.5)
    else:
        df = df.assign(diamHpupil=((df.pupil_postx - df.pupil_antx) ** 2 + (df.pupil_posty - df.pupil_anty) ** 2) ** 0.5,
                       diamVpupil=((df.pupil_ventx - df.pupil_dorx) ** 2 + (df.pupil_venty - df.pupil_dory) ** 2) ** 0.5,
                       eyeH=((df.eye_antx - df.eye_postx) ** 2 + (df.eye_anty - df.eye_posty) ** 2) ** 0.5)
    return df


def distanceMoved(vname, path, bodyparts, colorfig):
    """function returns the average distance in pixels from one frame to the next
    Parameters:
        vname (str): name of the video file to be worked on
        path (str): path where the h5 files are located with the video
        bodyparts (list): list of bodyparts of interest which are related eg. ['snoutL', 'snoutR', 'snoutTip']

    Return:
    	mainTemp (average distance difference in pixels)
    	save the meanFrame timeseries to numpy array

    Usage example:
        # distanceMoved('628shockPRO_cut_9809', 'C:/Users/Windows/Desktop/Lab/2020-06-06 - grantGavinVid', ['snoutL', 'snoutR', 'snoutTip'])
    """

    file = glob.glob(path+'/'+vname+'*.h5')[0]
    scorer = file.split(os.sep)[-1].split('.h5')[0].split(vname)[-1]
    df = pd.read_hdf(file, "df_with_missing")
    df = df[scorer]

    mainTemp=[]
    for i,j in enumerate(bodyparts):
        print(i,j)
        temp = pd.DataFrame(   list(((df[j]['x'] - df[j]['x'].shift()) ** 2 + (
                        df[j]['y'] - df[j]['y'].shift()) ** 2) ** 0.5),
                        columns=[j])
        mainTemp.append(temp)

    mainTemp = pd.concat(mainTemp, axis=1, sort=False)
    mainTemp = mainTemp.mean(axis=1)

    scalingfactor=2
    xdimIm=1335*scalingfactor
    ydimIm=182*scalingfactor
    custdpi=100*scalingfactor
    figure = plt.figure(frameon=False, figsize=(xdimIm / custdpi, ydimIm / custdpi))

    plt.plot(mainTemp, color=colorfig)
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    figure.savefig(path + '/'+ bodyparts[0] + '.png',
                   bbox_inches=Bbox([[0.0, 0.0], [xdimIm / custdpi, ydimIm / custdpi]]), pad_inches=0,
                   dpi=custdpi, transparent=True)
    plt.close('all')

#https://github.com/DeepLabCut/DeepLabCut/blob/6280ae027c550f64db696286b01a582dc196d4c4/deeplabcut/utils/make_labeled_video.py

def get_segment_indices(bodyparts2connect, all_bpts):
    bpts2connect = []
    for bpt1, bpt2 in bodyparts2connect:
        if bpt1 in all_bpts and bpt2 in all_bpts:
            bpts2connect.extend(
                zip(
                    *(
                        np.flatnonzero(all_bpts == bpt1),
                        np.flatnonzero(all_bpts == bpt2),
                    )
                )
            )
    return bpts2connect

def CreateVideo(
    clip,
    Dataframe,
    pcutoff,
    dotsize,
    colormap,
    bodyparts2plot,
    trailpoints,
    cropping,
    x1,
    x2,
    y1,
    y2,
    bodyparts2connect,
    skeleton_color,
    draw_skeleton,
    displaycropped,
    color_by,
):
    import argparse
    import os

    ####################################################
    # Dependencies
    ####################################################
    import os.path
    from pathlib import Path

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FFMpegWriter
    from skimage.draw import circle, line_aa
    from skimage.util import img_as_ubyte
    from tqdm import trange

    from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal, visualization
    from deeplabcut.utils.video_processor import (
        VideoProcessorCV as vp,
    )  # used to CreateVideo


    """ Creating individual frames with labeled body parts and making a video"""
    bpts = Dataframe.columns.get_level_values("bodyparts") # recover all the body parts (3 repetition for x y and likelihood)
    all_bpts = bpts.values[::3] # obtain only unique body parts by taking the 3rd only
    if draw_skeleton:
        color_for_skeleton = (
            np.array(mcolors.to_rgba(skeleton_color))[:3] * 255
        ).astype(np.uint8)
        # recode the bodyparts2connect into indices for df_x and df_y for speed
        bpts2connect = get_segment_indices(bodyparts2connect, all_bpts)

    if displaycropped:
        ny, nx = y2 - y1, x2 - x1
    else:
        ny, nx = clip.height(), clip.width()

    fps = clip.fps()
    nframes = len(Dataframe.index)
    duration = nframes / fps

    print(
        "Duration of video [s]: ",
        round(duration, 2),
        ", recorded with ",
        round(fps, 2),
        "fps!",
    )
    print("Overall # of frames: ", nframes, "with cropped frame dimensions: ", nx, ny)

    print("Generating frames and creating video.")
    df_x, df_y, df_likelihood = Dataframe.values.reshape((nframes, -1, 3)).T
    if cropping and not displaycropped:
        df_x += x1
        df_y += y1
    colorclass = plt.cm.ScalarMappable(cmap=colormap)

    bplist = bpts.unique().to_list()
    nbodyparts = len(bplist)
    if Dataframe.columns.nlevels == 3:
        nindividuals = 1
        map2bp = list(range(len(all_bpts)))
        map2id = [0 for _ in map2bp]
    else:
        nindividuals = len(Dataframe.columns.get_level_values("individuals").unique())
        map2bp = [bplist.index(bp) for bp in all_bpts]
        nbpts_per_ind = (
            Dataframe.groupby(level="individuals", axis=1).size().values // 3
        )
        map2id = []
        for i, j in enumerate(nbpts_per_ind):
            map2id.extend([i] * j)
    keep = np.flatnonzero(np.isin(all_bpts, bodyparts2plot))
    bpts2color = [(ind, map2bp[ind], map2id[ind]) for ind in keep]

    if color_by == "bodypart":
        C = colorclass.to_rgba(np.linspace(0, 1, nbodyparts))
    else:
        C = colorclass.to_rgba(np.linspace(0, 1, nindividuals))
    colors = (C[:, :3] * 255).astype(np.uint8)

    with np.errstate(invalid="ignore"):
        for index in trange(nframes):
            image = clip.load_frame()
            if displaycropped:
                image = image[y1:y2, x1:x2]

            # Draw the skeleton for specific bodyparts to be connected as specified in the config file
            if draw_skeleton:
                for bpt1, bpt2 in bpts2connect:
                    if np.all(df_likelihood[[bpt1, bpt2], index] > pcutoff) and not (
                        np.any(np.isnan(df_x[[bpt1, bpt2], index]))
                        or np.any(np.isnan(df_y[[bpt1, bpt2], index]))
                    ):
                        rr, cc, val = line_aa(
                            int(np.clip(df_y[bpt1, index], 0, ny - 1)),
                            int(np.clip(df_x[bpt1, index], 0, nx - 1)),
                            int(np.clip(df_y[bpt2, index], 1, ny - 1)),
                            int(np.clip(df_x[bpt2, index], 1, nx - 1)),
                        )
                        image[rr, cc] = color_for_skeleton

            for ind, num_bp, num_ind in bpts2color:
                if df_likelihood[ind, index] > pcutoff:
                    if color_by == "bodypart":
                        color = colors[num_bp]
                    else:
                        color = colors[num_ind]
                    if trailpoints > 0:
                        for k in range(1, min(trailpoints, index + 1)):
                            rr, cc = circle(
                                df_y[ind, index - k],
                                df_x[ind, index - k],
                                dotsize,
                                shape=(ny, nx),
                            )
                            image[rr, cc] = color
                    rr, cc = circle(
                        df_y[ind, index], df_x[ind, index], dotsize, shape=(ny, nx)
                    )
                    image[rr, cc] = color

            clip.save_frame(image)
    clip.close()

#########################
# create dict for color
#########################

# snout : '#DB6716'
# ear: '#DAC92B' ###
# body: '#59D7BA' ###
# paws : '#6F7AC8'
# pupilpro: '#970013'
# pupilpup: '#D367AD'

# plt.plot(mainTemp / np.max(mainTemp))
# plt.plot(df['ear_dor']['likelihood'])
#
# plt.plot(mainTemp.mean(axis=1))
#
#
# for i,j in enumerate(mainTemp.columns):
#     print(i,j)
#     plt.plot(mainTemp[str(j)]+i)

# # bodyparts2plot = ['snoutL', 'snoutR', 'snoutTip']
# # bodyparts2plot = ['ear_dor' , 'ear_postMax', 'ear_vent', 'ear_antDor', 'ear_antVent']
# # bodyparts = ['body_antDor', 'body_postMax', 'body_dor', 'body_post', 'body_postVent']
# # bodyparts = ['paw_postL', 'paw_antR', 'paw_antL']

path = r'C:\Users\Windows\Desktop\Lab\2020-06-06 - grantGavinVid'
vname= '628shockPRO_cut_9809' #'628shock_cut_9605' #628shockPRO_cut_9809
file = glob.glob(path + '/' + vname + '*.h5')[0]
scorer = file.split(os.sep)[-1].split('.h5')[0].split(vname)[-1]
Dataframe = pd.read_hdf(file, "df_with_missing")
skeleton_color='orange' # look into how to convert

config=r'Y:\MLGroup\DLCNetwork\shockTestTKCTom-TKC-2020-05-20-400images\config.yaml'
cfg = auxiliaryfunctions.read_config(config)
bodyparts2connect = cfg["skeleton"]
os.chdir(r'C:\Users\Windows\Desktop\Lab\2020-06-06 - grantGavinVid\file')

#########################################
##########################################


bpts = Dataframe.columns.get_level_values(
    "bodyparts")  # recover all the body parts (3 repetition for x y and likelihood)
all_bpts = bpts.values[::3]  # obtain only unique body parts by taking the 3rd only
bpts2connect = get_segment_indices(bodyparts2connect, all_bpts)
skeleton_color='orange' # look into how to convert

video=path+os.sep+'628shockPRO_cut_9809.mp4'
clip=vp(video)
fps = clip.fps()
nframes = len(Dataframe.index)
duration = nframes / fps


df_x, df_y, df_likelihood = Dataframe.values.reshape((nframes, -1, 3)).T
colorclass = plt.cm.ScalarMappable(cmap='jet')
bplist = bpts.unique().to_list()
nbodyparts = len(bplist)

nindividuals = 1
map2bp = list(range(len(all_bpts)))
map2id = [0 for _ in map2bp]

bodyparts2plot = ['snoutL', 'snoutR', 'snoutTip']
keep = np.flatnonzero(np.isin(all_bpts, bodyparts2plot))
bpts2color = [(ind, map2bp[ind], map2id[ind]) for ind in keep]
C = colorclass.to_rgba(np.linspace(0, 1, nindividuals))




















