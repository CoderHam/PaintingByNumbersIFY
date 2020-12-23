from copy import deepcopy
from collections import Counter
import numpy as np

def get_counts_of_vicinity_values(mat, x, y, xyrange):
    ymax, xmax = mat.shape 
    vicinVals = mat[max(x - xyrange, 0): min(x + xyrange, xmax), max(y - xyrange, 0): min(y + xyrange, ymax)].flatten()

    return Counter(vicinVals).most_common(1)[0][0]

def smoothen(mat):
    ymax, xmax = mat.shape 
    flat_mat = np.array([get_counts_of_vicinity_values(mat, x, y, 4) for x in range(0, xmax) for y in range(0, ymax)])

    return flat_mat.reshape(mat.shape)

def neighborsSame(mat, x, y):
    width = len(mat[0])
    height = len(mat)
    val = mat[y][x]
    xRel = [1, 0]
    yRel = [0, 1]
    for i in range(0,len(xRel)):
        xx = x + xRel[i]
        yy = y + yRel[i]
        if xx >= 0 and xx < width and yy >= 0 and yy < height:
            if mat[yy][xx] != val :
                return False
    return True

def outline(mat):
    ymax, xmax = mat.shape
    line_mat = np.array([0 if neighborsSame(mat, x, y) else 1 for y in range(0, ymax) for x in range(0, xmax)], dtype=np.uint8)

    return line_mat.reshape(mat.shape)

def getRegion(mat, cov, x, y):
    covered = deepcopy(cov)
    region = {'value': mat[y][x], 'x': [], 'y': [] }
    value = mat[y][x]

    queue = [[x, y]]
    while (len(queue) > 0):
    	coord = queue.pop()
    	if covered[coord[1]][coord[0]] == False and mat[coord[1]][coord[0]] == value:
            region['x'].append(coord[0])
            region['y'].append(coord[1])
            covered[coord[1]][coord[0]] = True
            if coord[0] > 0:
                queue.append([coord[0] - 1, coord[1]])
            if coord[0] < len(mat[0]) - 1:
                queue.append([coord[0] + 1, coord[1]])
            if coord[1] > 0:
                queue.append([coord[0], coord[1] - 1])
            if coord[1] < len(mat) - 1:
                queue.append([coord[0], coord[1] + 1])

    return region

def coverRegion(covered, region):
    for i in range(0,len(region['x'])):
	       covered[region['y'][i]][region['x'][i]] = True

def sameCount(mat, x, y, incX, incY):
    value = mat[y][x]
    count = -1
    while x >= 0 and x < len(mat[0]) and y >= 0 and y < len(mat) and mat[y][x] == value:
    	count+=1
    	x += incX
    	y += incY

    return count

def getLabelLoc(mat, region):
    bestI = 0
    best = 0
    for i in range(0,len(region['x'])):
    	goodness = sameCount(mat, region['x'][i], region['y'][i], -1, 0) * sameCount(mat, region['x'][i], region['y'][i], 1, 0) * sameCount(mat, region['x'][i], region['y'][i], 0, -1) * sameCount(mat, region['x'][i], region['y'][i], 0, 1)
    	if goodness > best:
            best = goodness
            bestI = i

    return {'value': region['value'], 'x': region['x'][bestI], 'y': region['y'][bestI] }

def getBelowValue(mat, region):
    x = region['x'][0]
    y = region['y'][0]
    print(region)
    while mat[y][x] == region['value']:
        print(mat[y][x])
        y+=1

    return mat[y][x]

def removeRegion(mat, region):
    if region['y'][0] > 0 :
	       newValue = mat[region['y'][0] - 1][region['x'][0]]
    else:
	       newValue = getBelowValue(mat, region)
    for i in range(0,len(region['x'])):
	       mat[region['y'][i]][region['x'][i]] = newValue


def getLabelLocs(mat):
    width = len(mat[0])
    height = len(mat)
    covered = [[False]*width]*height

    labelLocs = []
    for y in range(0,height):
        for x in range(0,width):
            if covered[y][x] == False:
                region = getRegion(mat, covered, x, y)
                coverRegion(covered, region)
            if len(region['x']) > 100:
                labelLocs.append(getLabelLoc(mat, region))
            else:
                removeRegion(mat, region)

    return labelLocs

def img_process(mat):

    # Smoothen image 
    smooth_mat = smoothen(mat)

    # Identify color regions
    # labelLocs = getLabelLocs(smooth_mat)
    # Drawing outline
    line_mat = outline(smooth_mat)

    return smooth_mat, line_mat
