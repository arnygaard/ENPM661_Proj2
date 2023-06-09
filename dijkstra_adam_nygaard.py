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

#defining action set
def moveUp(x,y):
    next_loc = [x, y+1]
    return next_loc
def moveDown(x,y):
    next_loc = [x, y-1]
    return next_loc
def moveLeft(x,y):
    next_loc = [x-1, y]
    return next_loc
def moveRight(x,y):
    next_loc = [x+1, y]
    return next_loc
def moveURight(x,y):
    next_loc = [x+1, y+1]
    return next_loc
def moveULeft(x,y):
    next_loc = [x-1, y+1]
    return next_loc
def moveDRight(x,y):
    next_loc = [x+1, y-1]
    return next_loc
def moveDLeft(x,y):
    next_loc = [x-1, y-1]
    return next_loc

#defining moveset
def moveset(map, start_node, end_node=0):
    next_node = []
    x, y = start_node
    if y < 249:
        next_node.append(moveUp(x,y))
    if y < 249 and x < 599:
        next_node.append(moveURight(x,y))
    if x < 599:
        next_node.append(moveRight(x,y))
    if y > 0 and x < 599:
        next_node.append(moveDRight(x,y))
    if y > 0:
        next_node.append(moveDown(x,y))
    if y > 0 and x > 0:
        next_node.append(moveDLeft(x,y))
    if x > 0:
        next_node.append(moveLeft(x,y))
    if x > 0 and y < 249:
        next_node.append(moveULeft(x,y))
    temp_node = next_node.copy()
    for i in range(0, len(next_node)):
        tx = next_node[i][0]
        ty = next_node[i][1]
        #making sure values stay within bounds
        if tx < 0 or ty < 0 or tx > 600 or ty > 250:
            temp_node.remove(next_node[i])
        if map[tx,ty,0] != 0:
            temp_node.remove(next_node[i])
    return temp_node

#create empty map
map_empty = np.zeros([600, 250, 3], dtype=np.uint8)
map_copy = map_empty.copy()

#5mm buffer as called for
buffer = 5

#boundary buffer

#creating square shape dimensions and assigning pixel values to map
x_square_start = 100
x_square_end = 150
y_square_start = 0
y_square_end = 100
for i in range(x_square_start, x_square_end):
    for j in range(y_square_start, y_square_end):
        map_empty[i,j,:] = [239,76,76]

#plotting buffer zone
for i in range(x_square_start-buffer, x_square_end+buffer):
    for j in range(y_square_start-buffer, y_square_end+buffer):
            if map_empty[i,j,0] != 239:
                map_empty[i,j,:] = [239,76,76]

x_square_start = 100
x_square_end = 150
y_square_start = 150
y_square_end = 249
for i in range(x_square_start, x_square_end):
    for j in range(y_square_start, y_square_end):
        map_empty[i,j,:] = [239,76,76]

#plotting buffer zone
for i in range(x_square_start-buffer, x_square_end+buffer):
    for j in range(y_square_start-buffer, y_square_end):
            if map_empty[i,j,0] != 239:
                map_empty[i,j,:] = [239,76,76]

#hexagon background
x_square_start = 230
x_square_end = 370
y_square_start = 45
y_square_end = 205
for i in range(x_square_start, x_square_end):
    for j in range(y_square_start, y_square_end):
        map_empty[i,j,:] = [239,76,76]


#cutting out corners of hexagon background to create hexagon. 5mm buffer accounted for
x_wall_start = 200
x_wall_end = 300
y_wall_start = 0
y_wall_end = 100
for i in range(x_wall_start, x_wall_end):
    for j in range(y_wall_start, y_wall_end):
        if -0.57*i < j and -0.57*i + 216 > j: 
            map_empty[i,j,:] = [0,0,0]

x_wall_start = 280
x_wall_end = 400
y_wall_start = 0
y_wall_end = 200
for i in range(x_wall_start, x_wall_end):
    for j in range(y_wall_start, y_wall_end):
        if 0.57*i - 125 > j and 0.57*i - 255 < j: 
            map_empty[i,j,:] = [0,0,0]

x_wall_start = 200
x_wall_end = 400
y_wall_start = 0
y_wall_end = 220
for i in range(x_wall_start, x_wall_end):
    for j in range(y_wall_start, y_wall_end):
        if 0.57*i + 80 > j and 0.57*i + 34 < j: 
            map_empty[i,j,:] = [0,0,0]

x_wall_start = 290
x_wall_end = 400
y_wall_start = 0
y_wall_end = 220
for i in range(x_wall_start, x_wall_end):
    for j in range(y_wall_start, y_wall_end):
        if -0.57*i + 375 < j and -0.57*i + 495 > j: 
            map_empty[i,j,:] = [0,0,0]


#triangle with 5mm buffer accounted for
x_wall_start = 460 - buffer
x_wall_end = 510 + buffer
y_wall_start = 125
y_wall_end = 240
for i in range(x_wall_start, x_wall_end):
    for j in range(y_wall_start, y_wall_end):
        if -7/4*i < j and -7/4*i + 1026 > j: 
            map_empty[i,j,:] = [239,76,76]

x_wall_start = 460 - buffer
x_wall_end = 510 + buffer
y_wall_start = 0
y_wall_end = 125
for i in range(x_wall_start, x_wall_end):
    for j in range(y_wall_start, y_wall_end):
        if 7/4*i - 776 < j and 7/4*i > j: 
            map_empty[i,j,:] = [239,76,76]

for i in range(0,600):
    for j in range(0,250):
        if i < 6:
            map_empty[i,j,:] = [239,76,76]
        if i > 594:
            map_empty[i,j,:] = [239,76,76]
        if j < 6:
            map_empty[i,j,:] = [239,76,76]
        if j > 244:
            map_empty[i,j,:] = [239,76,76]

#creating cost map of the map with included obstacles. -1 if within obstacle, 1E9 otherwise
dim = (600,250)
cost_map = np.zeros(dim)
print(cost_map.shape)
for i in range(0, 600):
    for j in range(0,250):
        if map_empty[i,j,0] == 0:
            cost_map[i,j] = 1E9
        else:
            cost_map[i,j] = -1

#calculate cost to come for each node
def CalcCost(child, parent):
    xc, yc = child
    xp, yp = parent
    lx = np.abs(xp-xc)
    ly =np.abs(yp-yc)
    cost = round(np.sqrt(lx**2 + ly**2),1)
    return cost

#backtracking function to find best path based on neighboring costs
#find lowest neighboring cost, move to that node and repeat
def backtrack(cost_map, start, goal):
    gx = goal[0]
    gy = goal[1]
    tx = 0
    ty = 0
    #ensuring values stay within bounds
    if goal == [600, 250]:
        gx = gx - 1
        gy = gy - 1
    sx = start[0]
    sy = start[1]
    path_nodes = []
    pnode = []
    lowest_cost = cost_map[gx,gy]
    pnode = [gx,gy]
    path_nodes.append(pnode)
    while pnode != start:
        path_nodes.append(pnode)
        gx = pnode[0]
        gy = pnode[1]
        if gx < 599:
            if cost_map[gx+1,gy] < lowest_cost and cost_map[gx+1,gy] >= 0 and gx < 600 and gy < 250:
                path_nodes.pop()
                lowest_cost = cost_map[gx+1,gy]
                pnode = [gx+1,gy]
                path_nodes.append(pnode)

        if gx < 599 and gy < 249:
            if cost_map[gx+1,gy+1] < lowest_cost and cost_map[gx+1,gy+1] >= 0 and gx < 600 and gy < 250:
                path_nodes.pop()
                lowest_cost = cost_map[gx+1,gy+1]
                pnode = [gx+1,gy+1]
                path_nodes.append(pnode)

        if gx < 599 and gy > 0:
            if cost_map[gx+1,gy-1] < lowest_cost and cost_map[gx+1,gy-1] >= 0 and gx < 600 and gy < 250:
                path_nodes.pop()
                lowest_cost = cost_map[gx+1,gy-1]
                pnode = [gx+1,gy-1]
                path_nodes.append(pnode)

        if gx > 0 and gy > 0:
            if cost_map[gx-1,gy-1] < lowest_cost and cost_map[gx-1,gy-1] >= 0 and gx < 600 and gy < 250:
                path_nodes.pop()
                lowest_cost = cost_map[gx-1,gy-1]
                pnode = [gx-1,gy-1]
                path_nodes.append(pnode)

        if gy > 0:
            if cost_map[gx,gy-1] < lowest_cost and cost_map[gx,gy-1] >= 0 and gx < 600 and gy < 250:
                path_nodes.pop()
                lowest_cost = cost_map[gx,gy-1]
                pnode = [gx,gy-1]
                path_nodes.append(pnode)
        if gy < 249:
            if cost_map[gx,gy+1] < lowest_cost and cost_map[gx,gy+1] >= 0 and gx < 600 and gy < 250:
                path_nodes.pop()
                lowest_cost = cost_map[gx,gy+1]
                pnode = [gx,gy+1]
                path_nodes.append(pnode)
        if gx > 0 and gy < 249:        
            if cost_map[gx-1,gy+1] < lowest_cost and cost_map[gx-1,gy+1] >= 0 and gx < 600 and gy < 250:
                path_nodes.pop()
                lowest_cost = cost_map[gx-1,gy+1]
                pnode = [gx-1,gy+1]
                path_nodes.append(pnode)
        if gx > 0:
            if cost_map[gx-1,gy] < lowest_cost and cost_map[gx-1,gy] >= 0 and gx < 600 and gy < 250:
                path_nodes.pop()
                lowest_cost = cost_map[gx-1,gy]
                pnode = [gx-1,gy]
                path_nodes.append(pnode)
    path_nodes.reverse()
    return path_nodes

#function to sort frames when processing output video
# numbers = re.compile(r"(\d+)")
# def numericalSort(value):
#     parts = numbers.split(value)
#     parts[1::2] = map(int, parts[1::2])
#     return parts

#assigning start and end points
print('Enter start node x:')
sx = int(input())
print('Enter start node y:')
sy = int(input())
print('Enter goal node x:')
gx = int(input())
print('Enter goal node y:')
gy = int(input())
print('Start Node:',sx, sy)
print('Goal Node:',gx, gy)
Xs = [sx, sy]

if gy >= 250:
    gy = 249
if gx >= 600:
    gx = 599
goal = [gx, gy]

#checking for goalpoint or start point in obstacle
if map_empty[gx,gy,0] != 0:
    print("goal lies in obstacle")
    sys.exit()
if map_empty[sx,sy,0] != 0:
    print("start lies in obstacle")
    sys.exit()

print(' ')
print('Print path coordinates after completion? (y/n)')
print_path = str(input())

#start timer
start_time = time.time()

#initializing lists and first cost of start node
OpenList = []
ClosedList = []
goal_state = 1
tcost = []
cost_map[Xs[0],Xs[1]] = 0

#begin djikstra's algorithm
OpenList.append(Xs)
while OpenList and goal_state != 0:

    Node_State_i = OpenList.pop(0)
    if Node_State_i == goal:
        print('SUCCESS')
        break
    
    testing_node = moveset(map_empty, Node_State_i)
    for item in testing_node:
        color = map_empty[item[0], item[1], 0]
        if color == 0:
            if item is not ClosedList:
                cost2come = CalcCost(item, Node_State_i) + cost_map[Node_State_i[0],Node_State_i[1]]
                if cost_map[item[0],item[1]] > cost2come:
                    cost_map[item[0],item[1]] = cost2come
                    OpenList.append(item)
                    ClosedList.append(item)
    testing_node = []

#find path from backtrack function
path = backtrack(cost_map, Xs, goal)

if print_path == 'y':
    print("path:", path)          

#plotting path
for node in path:
    x = node[0]
    y = node[1]
    map_empty[x,y,:] = [255, 255, 255]

#create final map
map_final = np.transpose(map_empty, (1, 0, 2))

#end timer and print
print("Djisktra's search took %s seconds" % (time.time() - start_time))

#plot map
plt.imshow(map_final)
plt.gca().invert_yaxis()
plt.show()

#writing individual frames to folder that show djikstra search
# ClosedList.append(goal)
# ims = []
# c = 0
# for item in ClosedList:
#     x = item[0]
#     y = item[1]
#     if x > 599:
#         x = 599
#     if y > 249:
#         y = 249
#     map_empty[x, y, :] = [119, 191, 88]
#     map_final = np.transpose(map_empty, (1, 0, 2))
#     map_final = np.flip(map_final,0)
#     cv.imwrite("images/Frame%d.jpg"%c, map_final)
#     c = c+1
#     if item == goal:
#         for node in path:
#             i = node[0]
#             j = node[1]
#             if i > 599:
#                 i = 599
#             if j > 249:
#                 j = 249
#             map_empty[i,j,:] = [255, 255, 255]
#             map_final = np.transpose(map_empty, (1, 0, 2))
#             map_final = np.flip(map_final,0)
#             cv.imwrite("images/Frame%d.jpg"%c, map_final)
#             c = c+1