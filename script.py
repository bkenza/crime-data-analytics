# -------------------------------------------------------
# Assignment 1
# Written by Kenza Boulisfane - 40043521
# For COMP 472 Section (JX) – Summer 2020
# --------------------------------------------------------

import numpy as np
import shapefile as shp
import matplotlib.pyplot as plt
from matplotlib import colors
import heapq
import time

# ----------------------------------
# VARIABLE CREATION/ INITIALISATION
# -----------------------------------

#  center of Montreal downtown area coordinates
coordinates = [(-73.59, 45.49), (-73.55, 45.49), (-73.55, 45.53), (-73.59, 45.53)]
coords = []

# Read crime data from file using Shapefile
shp_data = shp.Reader("Shape/crime_dt", encoding='ISO-8859-1')
print(shp_data)

# Create variables to be used in grid
min_xval, min_yval, max_xval, max_yval = shp_data.bbox

start_time = time.time()  # start timer

# Collect user input
box = float(input("Please enter a box size:"))  # size of each cell
threshold_input = float(input("Please enter a percent threshold to apply to the graph:"))  # arbitrary threshold
if threshold_input > 100 or threshold_input < 0:
    print("Invalid threshold. Please pick a value between 0% and 100%")
else:
    print("Thank you!")

# Set row and column size based on user input
row = np.math.ceil((max_xval - min_xval) / box)
col = np.math.ceil((max_yval - min_yval) / box)

# Create a grid based on the previously calculated row and column numbers
# Note: this matrix is filled with zeroes and it will be modified later in the program
graph = np.array([[0] * row] * col)

# In each little box, calculate the number of crimes
crimeArray = {}
numCrimes = 0

# import shape records
shapeRecords = shp_data.shapeRecords()

# Graph:
x_coord = []
y_coord = []


# --------------------------
# FUNCTIONS
# --------------------------


# Function to transpose grid
def grid_transpose(original_grid):
    transposed_grid = original_grid.transpose()
    return transposed_grid


# Function to flatten grid
def grid_flatten(transposed_grid):
    flattened_grid = transposed.flatten()
    return flattened_grid


# Function to sort arrays in reverse
def reverse_sort(array):
    array_sorted = sorted(array, reverse=True)
    return array_sorted


# Function to calculate the mean
def calculate_mean(sorted_array):
    mean_ = (np.mean(sorted_array))
    return mean_


# Function to calculate the median
def calculate_median(sorted_array):
    median_ = (np.median(sorted_array))
    return median_


# Function to calculate the standard deviation
def calculate_std(sorted_array):
    st_dev = (np.std(sorted_array))
    return st_dev


# Function to print the mean, median and standard deviation
def print_stats(mean_, median_, st_dev):
    print("The mean is: " + str(mean_))
    print("The median is: " + str(median_))
    print("The standard deviation is: " + str(st_dev))


# Function to sort the grid contents in descending order
def grid_reverse_sort(grid):
    grid = sorted(grid, reverse=True)
    return grid


# Function that converts threshold inputted by the user to percentile
def percentile_threshold(grid_desc, user_input):
    threshold_ = np.percentile(grid_desc, user_input)
    return threshold_


# This function is used to calculate the heuristic: Pythagoras's theorem to find the distance between two point
def heuristic(n, m):
    return np.sqrt((m[0] - n[0]) ** 2 + (m[1] - n[1]) ** 2)


''' A* algorithm: the following algorithm uses two main types of data structures: lists and dictionaries.

1. A list of 8 possible directions is created: up, down, left, right, diagonally up to the right, 
diagonally down to the right, diagonally up to the left and finally diagonally down to the left.

2. A closed list that will store the visited positions that do not have to be considered a second time.

3. An open list containing all the visited positions that are retained.

4. A dictionary containing all the parent positions of each "current" position respectfully.

5. A cumulative g value.

6. A cumulative f value based on the cost (g value) and the heuristic value.

7. Logic of the restrictions: to determine the colour of the current and adjacent squares, 
we compare their values to the chosen threshold. In other words, if the value at any position is greater than the 
threshold, then it is a high crime area and the colour is deduced to be yellow and vice-versa.

Additional comments are provided in the function below.'''


def astar_algorithm(array, start, destination):
    time1 = time.time()
    not_found = 'Sorry, no path was found for the chosen points!'  # string to return if no path is found

    # all the possible directions we can go
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    closed_list = []  # list of positions already visited that we don't have to pass through again
    open_list = []  # contains all the visited positions
    parent_positions = {}  # dictionary containing all the parent positions to the current position
    g = {start: 0}  # initialize the cost
    f = {start: heuristic(start, destination)}  # initialize f

    # add the first point to the open list
    heapq.heappush(open_list, (f[start], start))

    # as long as the open list is not empty, execute the loop
    while len(open_list) > 0:
        current_position = heapq.heappop(open_list)[1]

        #  we are at the final position
        if current_position == destination:
            print("Final destination reached!")
            time2 = time.time()
            total = time2 - time1
            print('The total A* time is: ' + str(total))  # total time of the algorithm
            traced_path = []

            # if the current position is in the parent dictionary (skipped in first iteration)
            while current_position in parent_positions:
                traced_path.append(current_position)  # add the current position to the path
                current_position = parent_positions[current_position]

            return traced_path  # stores the path aka all the valid visited positions

        closed_list.append(current_position)  # add the current position to the closed list

        # In this loop, we go through all the possible adjacent points using the 8 possible directions
        # and we calculate their respective g values
        # We also take into consideration the restrictions
        for n, m in directions:
            adjacent = current_position[0] + n, current_position[1] + m  # find the adjacent points

            # if current is purple & adjacent is purple
            if crimeArray.get(current_position, 0) <= threshold and crimeArray[adjacent[n], 0] <= threshold:

                # if diagonal
                if (n == 1 and m == 1) or (n == -1, m == 1) or (n == 1, m == -1) or (n == -1, m == -1):
                    possible_g = g[current_position] + heuristic(current_position, adjacent) + 1.5

                # if straight line
                elif (n == -1 and m == 0) or (n == 1, m == 0) or (n == 0, m == 1) or (n == 0, m == -1):
                    possible_g = g[current_position] + heuristic(current_position, adjacent)

            # if current is purple and adjacent is yellow
            elif (crimeArray.get(current_position, 0) < threshold and crimeArray[adjacent[n], 0 < threshold]) or (
                    crimeArray.get(current_position, 0) < threshold and crimeArray[adjacent[n], 0 < threshold]):

                # if straight line
                if (n == -1 and m == 0) or (n == 1, m == 0) or (n == 0, m == 1) or (n == 0, m == -1):
                    possible_g = g[current_position] + heuristic(current_position, adjacent) + 1.3
                else:
                    continue

            # if yellow next to yellow
            elif crimeArray.get(current_position, 0) > threshold and crimeArray[adjacent[n], 0 > threshold]:
                continue

            # if adjacent point is outside bounds, ignore and continue
            if 0 <= adjacent[0] < array.shape[0]:
                if 0 <= adjacent[1] < array.shape[1]:

                    if array[adjacent[0]][adjacent[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue

            # if adjacent point is in the closed list but its g score is less than the previously calculated,
            # ignore and continue the loop
            if adjacent in closed_list and possible_g >= g.get(adjacent, 0):
                continue

            # if the adjacent point is not in the open list or its g score is
            # smaller than the previously calculated one:
            # 1. add the current position (the adjacent point) to the parent positions
            # 2. update its g value to match the calculated one
            # 3. update the f value using the calculated g and the heuristic function
            # 3. add the adjacent point to the open list
            if adjacent not in [n[1] for n in open_list] or possible_g < g.get(adjacent, 0):
                parent_positions[adjacent] = current_position
                g[adjacent] = possible_g
                f[adjacent] = possible_g + heuristic(adjacent, destination)
                heapq.heappush(open_list, (f[adjacent], adjacent))

    time3 = time.time()
    nopathtime = time3 - time1
    print('The total A* time is: ' + str(nopathtime))
    return not_found


# Function to set the colours of the map
def set_map_colour(input_threshold):
    if input_threshold == 100:
        color = colors.ListedColormap(['blue', 'blue'])
        return color
    else:
        color = colors.ListedColormap(['purple', 'yellow'])
        return color


# -------------------------
# FUNCTION CALLS
# -------------------------

x_values = []
y_values = []
# Step 1: populate grid with crimes
for i in range(len(shapeRecords)):
    coords.append(shapeRecords[i].shape.__geo_interface__["coordinates"])
    x_values.append(shapeRecords[i].shape.__geo_interface__["coordinates"][0])
    y_values.append(shapeRecords[i].shape.__geo_interface__["coordinates"][1])
    x = int((shapeRecords[i].shape.__geo_interface__["coordinates"][0] - min_xval) / box)  # populate x array
    y = int((shapeRecords[i].shape.__geo_interface__["coordinates"][1] - min_yval) / box)  # populate y array
    graph[x][y] = graph[x][y] + 1  # increment by 1

    # Check if the added x and y are in the crimeArray
    if (x, y) in crimeArray.keys():
        crimeArray[x, y] = crimeArray[x, y] + 1
    else:
        crimeArray[x, y] = 1
    if numCrimes < crimeArray[x, y]:
        numCrimes = crimeArray[x, y]

# Step 2: transpose: interchange x and y values and then flatten array
transposed = grid_transpose(graph)
flattened = grid_flatten(transposed)
print(flattened)

# Step 3: calculate statistics
x_sorted = reverse_sort(flattened)
mean = calculate_mean(x_sorted)
median = calculate_median(x_sorted)
standard_dev = calculate_std(x_sorted)

# Step 4: print statistics
print_stats(mean, median, standard_dev)

# Step 5: sort grid contents in descending order
desc_grid = reverse_sort(flattened)

# Step 6: convert the threshold chosen by the user to percentile
threshold = percentile_threshold(desc_grid, threshold_input)
end_time = time.time()

# Step 7: set the colours and the bounds of the graph
colour = set_map_colour(threshold_input)
bounds = [0, threshold, 270]
norm = colors.BoundaryNorm(bounds, colour.N)
fig, ax = plt.subplots()

# Step 8: plot the grid with the threshold
plt.imshow(graph, aspect='auto', origin='lower', cmap=colour, norm=norm,
           extent=[min_xval, max_xval, min_yval, max_yval])  # set graph parameters
plt.title('Montreal Crime Analytics based on threshold')  # graph title
plt.xlabel('x coordinates')
plt.ylabel('y coordinates')
plt.colorbar().ax.set_title('threshold')  # add colour bar with label
plt.show()

# Step 9: plot the grid with the path

# Ask for user input regarding the points to use
startingX = int(input('Please enter an X starting coordinate: '))
startingY = int(input('Please enter a Y starting coordinate: '))
endingX = int(input('Please enter an X destination coordinate: '))
endingY = int(input('Please enter a  Y destination coordinate: '))

starting = (startingX, startingY)
ending = (endingX, endingY)

route = astar_algorithm(graph, starting, ending)

# only run this if a path is returned, if not, it's ignored
if isinstance(route, list):
    route = route + [starting]
    route = route[::-1]

print('The generated path is: ' + str(route))

x_coords = []
y_coords = []

# if a path is generated, add the x and y coordinates to separate lists to be able to plot them
if isinstance(route, list):
    for i in (range(0, len(route))):
        x_ = route[i][0]
        y_ = route[i][1]
        x_coords.append(x_)
        y_coords.append(y_)

fig, ax = plt.subplots()

# ax.hist2d(x_values, y_values, cmap=colour, norm=norm)
ax.imshow(graph, cmap=colour, norm=norm, origin='lower')
ax.scatter(starting[1], starting[0], marker="*", color="white", s=200)  # start
ax.scatter(ending[1], ending[0], marker="*", color="red", s=200)  # destination
ax.plot(y_coords, x_coords, color="black")
plt.show()
end_time = time.time()
total_time = end_time - start_time
print('Total execution time of the program : ' + str(total_time))

# Step 10: Let the user know that the program is done
print('The program has terminated. Thank you and goodbye!')
quit()
