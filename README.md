# COMP472
Summer 2020: Artificial Intelligence Assignment 1
Montreal Crime Data Analytics

---------------------------------------------------------
In this assignment, I have used the following libraries:
---------------------------------------------------------

1. numpy: to manipulate arrays and also calculate all the statistics related to the Montreal Crime Analysis program
2. shapefile: to read the contents of the crime data file
3. matplotlib.pyplot: to plot both graphs
4. matplotlib import colors: to put colours on the graph
5. heapq: priority queue used in the A* algorithm
6. time: to print the execution time of the program

-------------------------------------
Instructions on how run this program:
-------------------------------------

1. Install the libraries listed above
2. Import the shape folder and add it to the project directory
3. Run the program
4. Input valid values when prompted (in the command line)

------------------------------
Notes about the A* algorithm:
------------------------------

A* algorithm: the implemented algorithm uses two main types of data structures: lists and dictionaries.

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

