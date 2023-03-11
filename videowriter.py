import re
import matplotlib
matplotlib.use("Agg")
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sys
import glob
from IPython import display
import os

#used from OpenCV Tuturial
numbers = re.compile(r"(\d+)")
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

c = 0
img_array = []
for filename in sorted(glob.glob(r"C:/Users/Adam's PC/OneDrive/Desktop/661/images/*.jpg"), key=numericalSort): #Change directory to your directory
    #only appending every 5th frame otherwise computer crashes
    if int(filename[-6:][1]) % 5 == 0:
        img = cv.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

#output video
out = cv.VideoWriter('results2.avi',cv.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()