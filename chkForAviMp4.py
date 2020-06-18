import os
import time
import subprocess

# input the file of intrest

def chkForAviMp4():
    """ this function enable the conversion of a file when the file is not updated anymore
    for example in the context of the highspeed seq file is converted to an avi. When the conversion
    is completed the size of the file will be static then the compression from avi to mp4 will occur after
    x amount of time from the finished conversion (by default 15 minutes)

    Args:
        filename (str): name of the file to be checked and converted
        checkEveryMin (int): optional by default check the file status every 15 minutes
    Returns:
        save a newFile
    """
    filename = input("Drag and drop the file to be converted: ")
    checkEveryMin = input("How often should the file progresss be monitored (in min - 15 recommanded): ")
    checkEveryMin = float(checkEveryMin)
    pathDir=os.path.dirname(filename) # extract the folder that contains the files
    baseFile=os.path.basename(filename).split('.')[0]
    newFile=pathDir+os.sep+baseFile+'_HS.mp4'
    os.chdir(pathDir) # place the files into this directory

    prevSize = 0
    while prevSize < os.stat(filename).st_size:
        prevSize=os.stat(filename).st_size
        print('still converting - current file size: ', str(prevSize))
        time.sleep(checkEveryMin*60)

    else:
        subprocess.call('ffmpeg -i ' + filename + ' -vcodec libx264 -crf 20 ' + newFile, shell=True)
        print('done new file is saved as :' + newFile)


chkForAviMp4()
