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
        if tx < 0 or ty < 0 or tx > 600 or ty > 250:
            temp_node.remove(next_node[i])
        if map[tx,ty,0] != 0:
            temp_node.remove(next_node[i])
    return temp_node

map_empty = np.zeros([600, 250, 3], dtype=np.uint8)
map_copy = map_empty.copy()

buffer = 5
x_square_start = 450
x_square_end = 550
y_square_start = 50
y_square_end = 230
for i in range(x_square_start, x_square_end):
    for j in range(y_square_start, y_square_end):
        map_empty[i,j,:] = [239,76,76]

for i in range(x_square_start-buffer, x_square_end+buffer):
    for j in range(y_square_start-buffer, y_square_end+buffer):
            if map_empty[i,j,0] != 239:
                map_empty[i,j,:] = [200,16,16]
        

x_hc = 300
hc_r = 100
y_hc = 120
for i in range(x_hc - hc_r, x_hc + hc_r):
    for j in range(y_hc - hc_r, y_hc + hc_r):
        if (i - x_hc) **2 + (j - y_hc)**2 <= hc_r**2 and j > y_hc:
            map_empty[i,j,:] = [239,76,76]

for i in range(x_hc - hc_r - buffer, x_hc + hc_r + buffer):
    for j in range(y_hc - hc_r - buffer, y_hc + hc_r + buffer):
        if (i - x_hc) **2 + (j - y_hc)**2 <= (hc_r+buffer)**2 and j > y_hc - 5 and map_empty[i,j,0] != 239:
            map_empty[i,j,:] = [200,16,16]

x_wall_start = 80
x_wall_end = 300
y_wall_start = 0
y_wall_end = 100

for i in range(x_wall_start, x_wall_end):
    for j in range(y_wall_start, y_wall_end):
        if 0.25*i < j and 0.25*i + 25 > j: 
            map_empty[i,j,:] = [239,76,76]

for i in range(x_wall_start - buffer, x_wall_end + buffer):
    for j in range(y_wall_start - buffer, y_wall_end + buffer):
        if 0.25*i < j+buffer and 0.25*i + 25 > j-buffer and map_empty[i,j,0] != 239: 
            map_empty[i,j,:] = [200,16,16]

dim = (600,250)
cost_map = np.zeros(dim)
print(cost_map.shape)
for i in range(0, 600):
    for j in range(0,250):
        if map_empty[i,j,0] == 0:
            cost_map[i,j] = 1E9
        else:
            cost_map[i,j] = -1

def CalcCost(child, parent):
    xc, yc = child
    xp, yp = parent
    lx = np.abs(xp-xc)
    ly =np.abs(yp-yc)
    cost = round(np.sqrt(lx**2 + ly**2),1)
    return cost

def backtrack(cost_map, start, goal):
    gx = goal[0]
    gy = goal[1]
    if goal == [600, 250]:
        gx = gx - 1
        gy = gy - 1
    sx = start[0]
    sy = start[1]
    path_nodes = []
    pnode = []
    goal_cost = cost_map[gx,gy]
    lowest_cost = cost_map[gx,gy]
    pnode = [gx,gy]
    path_nodes.append(pnode)
    while pnode != start:
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

# numbers = re.compile(r"(\d+)")
# def numericalSort(value):
#     parts = numbers.split(value)
#     parts[1::2] = map(int, parts[1::2])
#     return parts

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
goal = [gx, gy]

if map_empty[gx,gy,0] != 0:
    print("goal lies in obstacle")
    sys.exit()
if map_empty[sx,sy,0] != 0:
    print("start lies in obstacle")
    sys.exit()

OpenList = []
ClosedList = []
# Xs = [0,0]
# goal = [10, 10]
goal_state = 1
tcost = []
cost_map[Xs[0],Xs[1]] = 0

OpenList.append(Xs)
while OpenList and goal_state != 0:

    Node_State_i = OpenList.pop(0)
    #ClosedList.append(Node_State_i)
    if Node_State_i == goal:
        #backtrack
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
print("done")
path = backtrack(cost_map, Xs, goal)
print("path:", path)          

for node in path:
    x = node[0]
    y = node[1]
    map_empty[x,y,:] = [255, 255, 255]

map = np.transpose(map_empty, (1, 0, 2))

plt.imshow(map)
plt.gca().invert_yaxis()
plt.show()

#ClosedList.append(goal)
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
# img_array = []
# #path = r"C:\Users\Adam's PC\OneDrive\Desktop\661\images*.jpg"
# for filename in sorted(glob.glob(r"C:/Users/Adam's PC/OneDrive/Desktop/661/images/*.jpg"), key=numericalSort):
#     img = cv.imread(filename)
#     height, width, layers = img.shape
#     size = (width,height)
#     img_array.append(img)

 
# out = cv.VideoWriter('results.avi',cv.VideoWriter_fourcc(*'DIVX'), 15, size)
 
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()

# print(ims[1][1])
# plt.imshow(ims[5])
# plt.show()
# cv.imshow("a",ims[1])
# cv.waitKey(0)
# out = cv.VideoWriter('results.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, size)
# for i in range(len(ims)):
#     out.write(ims[i])
# out.release()

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat=False)
# plt.suptitle('Adam Nygaard             Djisktra Search')
# plt.gca().invert_yaxis()
# ani.save('results.mp4', writer=writer)

#ani.save('animation.mp4', progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'))

#plt.show()


# ClosedList.append(goal)
# fig, ax = plt.subplots()
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
#     map = np.transpose(map_empty, (1, 0, 2))
#     im = ax.imshow(map, animated=True)
#     ims.append([im])
#     if item == goal:
#         for node in path:
#             i = node[0]
#             j = node[1]
#             if i > 599:
#                 i = 599
#             if j > 249:
#                 j = 249
#             map_empty[i,j,:] = [255, 255, 255]
#             map = np.transpose(map_empty, (1, 0, 2))
#             im = ax.imshow(map, animated=True)
#             ims.append([im])



