
# Maze generator and solver

This is a modification of the [pymaze](https://github.com/jostbr/pymaze) project publicly available on GitHub used for my Intro to AI course at Sungkyunkwan University. The modifications include rewriting search algorithms such as UCS, A*, and Q-learning.

## Additions:

For the purposes of this assignment, there have been some modifications to the source code.
The src and tests folders are now located inside the /examples/ directory. The /img/ folder contains saved visuals of maze-solving algorithms.

Python 3.10 or higher is required.

Firstly, ensure you have matplotlib==3.5.1 or higher installed.

`pip install -r requirements.txt`

To run the code, type the command:

`python examples/quick_start.py`

3 random mazes will be generated. As the code executes, the following visuals will be displayed for each maze:

    1. The layout of the maze
    2. An animation of the Q-learning algorithm navigating the maze after the first episode
    3. The resulting solution produced by Q-learning after the first episode
    2. An animation of the Q-learning algorithm navigating the maze after the last episode
    3. The resulting solution produced by Q-learning after the last episode
The solutions are then saved into the /img/ folder.
In the solution images, the red path represents the optimal route determined by the algorithm. The green/grey dots indicate other cells that the algorithm explored. Likewise, 'Cost' represents the number of moves required in the optimal path, while 'Steps' indicates the total number of cells visited by the algorithm. For Q-learning the optimal path is not shown, just the cells visited.

Only attempt to run examples/quick_start.py. Other python files are not guaranteed to execute properly.

Files modified:

	1. examples/quick_start.py
	2. examples/src/solver.py
	3. examples/src/maze.py
	4. examples/src/maze_manager.py
    (The following files were edited in UCS/A* assignment)
	5. examples/src/maze_viz.py
	6. examples/src/cell.py


# Original README
Python scripts for generating random solvable mazes using the depth-first search and recursive backtracking algorithms. The code also implements a recursive backtracking pathfinding algorithm for solving the generated mazes. Here is an example of a generated maze and its computed solution.  


Both the generator and solver algorithm uses recursive backtracking and here an example of the latter can be seen. Cells indicated in light orange are part of the backtracking. The algorithms works by moving randomly from a cell to one of its unvisited neighbours. If the search reaches cell which have no unvisited neighbours, the search backtracks until it moves to a cell with unvisited neighbours. The generator algorithm is heavily inspired by the pseudo code provided by [Wikipedia](https://en.wikipedia.org/wiki/Maze_generation_algorithm). The main difference between the generator and solver algorithms are in the fact that, when solving the maze, one has to take into account not being able to move through walls. And thus proper pathfinding needs to be implemented. There's also implemented an enhanced version of the solver algorithm which moves not to a random neighbour, but moves to the neighbour that minimizes the distance sqrt(x^2 + y^2) to the exit cell (final destination).


## Quick Use Guide
The first step is to install the dependencies by opening the terminal, navigating to 
the MazeGenerator directory, and running

`pip install -r requirements.txt`

Next, run the `quick_start` python example under the examples directory. If this ran without any errors,
you should be fine to create your own program. Use the format outlined in quick_start, or use
another example as a template.

The process for creating and solving a maze follows.

    1. Create a maze manager
    2. Add a maze to the manager
    3. Solve the maze
    4. Optionally visualize the results


An example of using the library with different options is shown below.


```python

from __future__ import absolute_import
from src.maze_manager import MazeManager
from src.maze import Maze


if __name__ == "__main__":

    # The easiest way to use the library is through the Manager class. It acts as the glue between
    # The visualization, solver, and maze classes. Mazes inside the manager have unique ids that we use
    # to specify particular mazes.
    manager = MazeManager()

    # We can add mazes to the manager two different ways.
    # The first way, we specify the maze dimensions. The maze that is created gets returned back to you.
    maze = manager.add_maze(10, 10)

    # The second way is by creating a maze, and then adding it to the manager. Doing this will require you to add
    # from src.maze import Maze
    # to your imports. Because the ids need to be unique, the manager will ensure this happens. It may change the
    # id of the maze that was passed in, so we assign it to the return value to make sure we're using the updated maze.
    maze2 = Maze(10, 10)
    maze2 = manager.add_existing_maze(maze2)

    # Once we have a maze in the manager, we can tell the manager to solve it with a particular algorithm.
    #manager.solve_maze(maze.id, "BreadthFirst")
    #manager.solve_maze(maze.id, "BiDirectional")
    manager.solve_maze(maze.id, "DepthFirstBacktracker")

    # If we want to save the maze & maze solution images along with their animations, we need to let the manager know.
    manager.set_filename("myFileName")

    # To see the unsolved maze, call
    manager.show_maze(maze.id)

    # You can also set the size of the cell by passing show_maze's second argument. The default is 1.
    # manager.show_maze(maze.id, 2)

    # To show an animation of how the maze was generated, use the following line
    manager.show_generation_animation(maze.id)

    # You can also see an animation of how the solver went about finding the end
    manager.show_solution_animation(maze.id)

    # Finally, you can show an image of the maze with the solution path overlaid. All of these display
    # functions will save the figure if MazeManager::set_filename has been set.
    manager.show_solution(maze.id)
```




## Developer's Guide

### Source Layout
* /src/   Holds the source code (modules) needed to run MazeGenerator.
* /tests/ Holds the unit tests that test the code in /src/
* /examples/ Example files that demonstrate how to use the library. 


### Class Overview
* The`Maze` class. This class provides helper functions to easily manipulate the cells. It can be thought of as being a grid of Cells
* The `Cell` class is used to keep track of walls, and is what makes up the list.
* The `Visualizer` class is responsible for handling the generation, display, and saving of animations and grid images. It can be interacted with directly, or controlled thought the `MazeManager` class.
* The `Solve` class. All solution methods are derived from this class. 
* The `MazeManager` class acts as the glue, bridging the `Visualizer`, `Maze`, and `Solve` classes together.
