import subprocess
import os 
import glob 
import pandas as pd 
import numpy as np
from tqdm import tqdm # used to measure remaining time progress

def convert(seconds):

    """Function to convert the time input in seconds to format HH:MM:SS
    
    Return:
        output in HH:MM:SS

    Parameters:
        seconds (int): simulated number of group
    """
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%02d:%02d:%02d" % (hour, minutes, seconds) 

def videoChopping(folder, timeWindow=[4,4], fps=30, vidFormat='mp4'):
    """Function to take input of file list and corresponding time point of interest
    around which the video should be cropped in time
    
    Return:
        mp4 files which are cut out around the interval of interest in an output folder
        of the slected folder for the function

    Parameters:
        folder (str): string name of the folder with which contains video and numpy array of timestamps of interest
                      the timestamps default are in stored by their frame index in a numpy array
                      NOTE: avoid folder name with spaces
        vidFormat (str): format the type of video to be converted (mp4, avi, etc.)
        timeWindow (list, int): integer list of time surrounding the frame of interest expressed in seconds
        fps (int): frame per seconds of the video acquisition
    """

    # recover the video data
    path=folder
    os.makedirs(path+'/output', exist_ok=True)
    video=glob.glob(path+'/*.'+vidFormat)
    aid=video[0].split(os.sep)[-1].split('.')[0]

    # recover the time data
    temptime=np.load(glob.glob(path+'/*.npy')[0])

    for ii in temptime:
        print(ii)
        # convert the start frame and stop frame around the interval in seconds
        jStart=convert((ii-timeWindow[0]*fps)/fps)
        jStop=convert((ii+timeWindow[1]*fps)/fps)
        print(jStart, jStop)
        subprocess.call('ffmpeg -i '+video[0]+' -ss '+jStart+ ' -to '+ jStop + ' ' + path+'/output/'+aid+'_cut_'+str(ii)+'.mp4', shell=True)  

def fileToConcat(folder):
    """Function to create a file list of video to concatenate
    
    Return:
        a text file that contains all the video to be concatenated and can be interpreted by ffmpeg

    Parameters:
        folder (str): string name of the folder with which contains chopped/cropped in time video to be concatenated
    """

    filesList=glob.glob(folder+'/*cut*.mp4')
    filesList=['file \'' + x+'\'' for x in filesList]
    os.chdir(folder)
    with open('concatList.txt', 'w') as file_handler:
        for item in filesList:
            file_handler.write("{}\n".format(item))

def videoConcat(folder, outputName='MyOutput'):
    """Function to concatenate a series of video together based on a file list
    
    Return:
        a text file that contains all the video to be concatenated and can be interpreted by ffmpeg

    Parameters:
        folder (str): string name of the folder with which contains chopped/cropped in time video to be concatenated
    """
    concatList=glob.glob(folder+'/*.txt')[0].split(os.sep)[-1]
    os.chdir(folder)
    subprocess.call('ffmpeg -f concat -safe 0 -i '+concatList+' -c copy '+outputName+'.mp4', shell=True)


############################
# USAGE EXAMPLE
############################

folder='C:/Users/Windows/Desktop/Newfolder'
vidFormat='mp4'
timeWindow=[4,4]
fps=30

videoChopping(folder, timeWindow=[4,30], fps=30, vidFormat='mp4')
fileList(folder+'/output')
videoConcat(folder+'/output')