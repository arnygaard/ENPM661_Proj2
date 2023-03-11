# ENPM661_Proj2
Libraries needed to run code:

import matplotlib
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sys
import glob
from IPython import display
import re


In order to run the algorithm correctly, first make sure to run the code in a compiler. I used VSC. 

When running the code, you will be asked to input start and goal points. Make sure to enter each number seperately, as stated by the instructions in the terminal. Only enter one coordinate at a time, as the code will not accept xy coordinate pairs for start and goal nodes.

If you want to have the path printed out node-by-node, input y when asked.

the source code will display the optimal path in the obstacle space by default.

If you input coordinates inside an obstacle, the program will output a failure message and will need to be rerun.
