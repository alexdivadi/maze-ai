import time
import random
import logging
from enum import Enum
from random import choice, random
from src.maze import Maze, Action

logging.basicConfig(level=logging.DEBUG)

def index_list(lst, val):
    return [i for i, x in enumerate(lst) if x==val]

class Solver(object):
    """Base class for solution methods.
    Every new solution method should override the solve method.

    Attributes:
        maze (list): The maze which is being solved.
        neighbor_method:
        quiet_mode: When enabled, information is not outputted to the console

    """

    def __init__(self, maze, quiet_mode, neighbor_method):
        logging.debug("Class Solver ctor called")

        self.maze = maze
        self.neighbor_method = neighbor_method
        self.name = ""
        self.quiet_mode = quiet_mode

    def solve(self):
        logging.debug('Class: Solver solve called')
        raise NotImplementedError

    def get_name(self):
        logging.debug('Class Solver get_name called')
        raise self.name

    def get_path(self):
        logging.debug('Class Solver get_path called')
        return self.path

    def get_optimal_path(self):
        logging.debug('Class Solver get_optimal_path called')
        self.maze.optimal_path = list()
        k_curr, l_curr = self.maze.exit_coor
        cell_curr = self.maze.grid[k_curr][l_curr]
        while cell_curr:
            self.maze.optimal_path.append(((cell_curr.row, cell_curr.col), False))
            self.maze.optimal_path_cost += 1
            cell_curr = cell_curr.parent
        self.maze.optimal_path.reverse()

class QLearning(Solver):

    class Action(Enum):
        """ List of possible actions for model (also Q-table index) """
        NORTH = 0
        EAST = 1
        SOUTH = 2
        WEST = 3

    def __init__(self, maze, quiet_mode=False, neighbor_method="brute-force", alpha=1, gamma=1, epsilon=0, max_iter=1000):
        logging.debug('Class Q-learning called')

        self.name = "Q Learning"
        self.learn_rate = alpha # rate at which model learns
        self.discount_rate = gamma # rate at which model values future state info
        self.explore_rate = epsilon # rate at which model randomly explores
        self.max_iterations = max_iter # maximum iterations for algorithm

        super().__init__(maze, neighbor_method, quiet_mode)

    def choose_action(self, q_table, state):
        """3 Criteria for choosing an action:
            1. If every action in the table is 0 (initial value), choose a random action
            2. With probability epsilon, choose a random action
            3. With probability 1-epsilon, choose the best action given q-table
               - If there is more than one best action (tie) choose tiebreaker randomly """
        actions = q_table[self.state_index(state, self.maze.num_cols)]
        if random() < self.explore_rate:
            #print("random!")
            return actions.index(choice(actions))
        return self.get_best_action(q_table, state)

    def try_action(self, state, a, maze):
        """Check if action is possible. Return next-state if in-bounds/no wall"""
        next_state = self.get_next_state(state, a)
        if maze.is_wall(state, next_state):
            return state
        return next_state

    def update_q(self, q_table, state, a, maze):
        """Performs the Q-update function"""
        next_state = self.get_next_state(state, a)
        i = self.state_index(state, maze.num_cols)
        i_next = self.state_index(next_state, maze.num_cols)
        # if the next state is impossible, update q-value without checking next_state
        if self.get_reward(state, next_state, maze) == -1:
            q_table[i][a] += self.learn_rate * (self.get_reward(state, next_state, maze) - q_table[i][a])
        else:
            q_table[i][a] += self.learn_rate * (
                                self.get_reward(state, next_state, maze) \
                                + self.discount_rate \
                                * q_table[i_next][self.get_best_action(q_table, next_state)] \
                                - q_table[i][a]
                            )

    def state_index(self, state, cols):
        """Converts state to index in Q-table"""
        return state[0] * cols + state[1]

    def get_next_state(self, state, a):
        """Gets next state from action"""
        next_state = list(state)
        match a:
            case Action.NORTH.value:
                next_state[0] += 1
            case Action.EAST.value:
                next_state[1] += 1
            case Action.SOUTH.value:
                next_state[0] -= 1
            case Action.WEST.value:
                next_state[1] -= 1
        return tuple(next_state)

    def get_best_action(self, q_table, state):
        """Finds action with maximum q-value"""
        actions = q_table[self.state_index(state, self.maze.num_cols)]
        #print(actions)
        return choice(index_list(actions, sorted(actions, reverse=True)[0]))

    def get_reward(self, current_state, next_state, maze):
        """Returns reward for state and action. Given by assignment."""
        if maze.is_wall(current_state, next_state):
            reward = -1
        elif next_state == maze.exit_coor:
            reward = 100
        else:
            reward = -0.1
        return reward
        
    def solve(self):

        logging.debug("Class Q-learning solve called")

        # q-table is a 2D list: q_table[state][action]
        q_table = self.maze.q_table
        current_state = self.maze.entry_coor  # initial state
        path = list()  # To track path of solution cell coordinates
        iterations = 0 # track iterations
        action = -1 # initialize action

        #print("\nSolving the maze with Q-learning search...")
        time_start = time.time()

        while iterations < self.max_iterations:  
            (k_curr, l_curr) = current_state
            path.append(((k_curr, l_curr), False))  # Append current cell to total search path

            if current_state == self.maze.exit_coor:  # Exit if current cell is exit cell
                if not self.quiet_mode:
                    print("Number of moves performed: {}".format(len(path)))
                    print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))
                    #print(q_table)
                return path
            
            #while True:
            action = self.choose_action(q_table, current_state) # choose action
            #print(current_state, q_table[self.state_index(current_state, self.maze.num_cols)], action)
            self.update_q(q_table, current_state, action, self.maze) # update q_table
                #if self.get_reward(current_state, self.get_next_state(current_state, action), self.maze) != -1:
                #    break
            current_state = self.try_action(current_state, action, self.maze) # update to next state

            iterations += 1 # increase iterations (steps) by one

        # Reach here if max iterations passed
        logging.debug("Class Q-learning leaving solve: Max iterations reached")
        if not self.quiet_mode:
            print("Number of moves performed: {}".format(len(path)))
            print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))
            #print(q_table)
        return path

class UCS(Solver):

    def __init__(self, maze, quiet_mode=False, neighbor_method="brute-force"):
        logging.debug('Class UCS ctor called')

        self.name = "Uniform Cost Search"
        super().__init__(maze, neighbor_method, quiet_mode)

    def solve(self):

        logging.debug("Class UCS solve called")
        cost = 0
        current_level = [(self.maze.entry_coor, cost)]  # Stack of cells at current level of search
        path = list()  # To track path of solution cell coordinates

        print("\nSolving the maze with UC search...")
        time_start = time.time()

        while True:  # Loop until return statement is encountered
            next_level = list()
            cost += 1

            while current_level:  # While still cells left to search on current level
                (k_curr, l_curr), _ = current_level.pop(0)  # Search one cell on the current level
                self.maze.grid[k_curr][l_curr].visited = True  # Mark current cell as visited
                path.append(((k_curr, l_curr), False))  # Append current cell to total search path
                

                if (k_curr, l_curr) == self.maze.exit_coor:  # Exit if current cell is exit cell
                    if not self.quiet_mode:
                        print("Number of moves performed: {}".format(len(path)))
                        print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))
                    return path

                neighbour_coors = self.maze.find_neighbours(k_curr, l_curr)  # Find neighbour indicies
                neighbour_coors = self.maze.validate_neighbours_solve(neighbour_coors, k_curr,
                                                                  l_curr, self.maze.exit_coor[0],
                                                                  self.maze.exit_coor[1], method="brute-force")

                if neighbour_coors is not None:
                    for coor in neighbour_coors:
                        next_level.append(coor)  # Add all existing real neighbours to next search level

            for cell in next_level:
                current_level.append((cell, cost))  # Update current_level list with cells for nex search level
        logging.debug("Class UCS leaving solve")

class AStar(Solver):

    def __init__(self, maze, quiet_mode=False, neighbor_method="brute-force"):
        logging.debug('Class A_Star ctor called')

        self.name = "A* Search"
        super().__init__(maze, neighbor_method, quiet_mode)

    def solve(self):

        logging.debug("Class A_Star solve called")
        current_level = [(self.maze.entry_coor, 0)]  # Stack of cells at current level of search
        self.maze.grid[self.maze.entry_coor[0]][self.maze.entry_coor[1]].cost = 0 # initial cost is 0
        path = list()  # To track path of solution cell coordinates

        print("\nSolving the maze with A* search...")
        time_start = time.time()

        while True:  # Loop until return statement is encountered

            while current_level:  # While still cells left to search on current level
                # pop min value from queue
                (k_curr, l_curr), _ = current_level.pop(current_level.index(min(current_level, key=lambda ls: ls[1]))) 

                self.maze.grid[k_curr][l_curr].visited = True  # Mark current cell as visited
                path.append(((k_curr, l_curr), False))  # Append current cell to total search path
                
                g = self.maze.grid[k_curr][l_curr].cost

                if (k_curr, l_curr) == self.maze.exit_coor:  # Exit if current cell is exit cell
                    if not self.quiet_mode:
                        print("Number of moves performed: {}".format(len(path)))
                        print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))
                    return path

                neighbour_coors = self.maze.find_neighbours(k_curr, l_curr)  # Find neighbour indicies
                neighbour_coors = self.maze.validate_neighbours_solve(neighbour_coors, k_curr,
                                                                  l_curr, self.maze.exit_coor[0],
                                                                  self.maze.exit_coor[1], method="brute-force")

                if neighbour_coors is not None:
                    for coor in neighbour_coors:
                        # get euclidean distance of coor and add to queue
                        self.maze.grid[coor[0]][coor[1]].cost = g + 1 # get g score from previous cell score + 1
                        h = Maze.euclidean_distance(coor[0], coor[1], self.maze.exit_coor[0], self.maze.exit_coor[1])
                        current_level.append((coor, h + g + 1))  # Add all existing real neighbours to next search level

        logging.debug("Class AStar leaving solve")

class BreadthFirst(Solver):

    def __init__(self, maze, quiet_mode=False, neighbor_method="fancy"):
        logging.debug('Class BreadthFirst ctor called')

        self.name = "Breadth First Recursive"
        super().__init__(maze, neighbor_method, quiet_mode)

    def solve(self):

        """Function that implements the breadth-first algorithm for solving the maze. This means that
                for each iteration in the outer loop, the search visits one cell in all possible branches. Then
                moves on to the next level of cells in each branch to continue the search."""

        logging.debug("Class BreadthFirst solve called")
        current_level = [self.maze.entry_coor]  # Stack of cells at current level of search
        path = list()  # To track path of solution cell coordinates

        print("\nSolving the maze with breadth-first search...")
        time_start = time.time()

        while True:  # Loop until return statement is encountered
            next_level = list()

            while current_level:  # While still cells left to search on current level
                print(current_level)
                k_curr, l_curr = current_level.pop(0)  # Search one cell on the current level
                self.maze.grid[k_curr][l_curr].visited = True  # Mark current cell as visited
                path.append(((k_curr, l_curr), False))  # Append current cell to total search path
                

                if (k_curr, l_curr) == self.maze.exit_coor:  # Exit if current cell is exit cell
                    if not self.quiet_mode:
                        print("Number of moves performed: {}".format(len(path)))
                        print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))
                    return path

                neighbour_coors = self.maze.find_neighbours(k_curr, l_curr)  # Find neighbour indicies
                neighbour_coors = self.maze.validate_neighbours_solve(neighbour_coors, k_curr,
                                                                  l_curr, self.maze.exit_coor[0],
                                                                  self.maze.exit_coor[1], self.neighbor_method)

                if neighbour_coors is not None:
                    for coor in neighbour_coors:
                        next_level.append(coor)  # Add all existing real neighbours to next search level

            for cell in next_level:
                current_level.append(cell)  # Update current_level list with cells for nex search level
        logging.debug("Class BreadthFirst leaving solve")


class BiDirectional(Solver):

    def __init__(self, maze, quiet_mode=False, neighbor_method="fancy"):
        logging.debug('Class BiDirectional ctor called')

        super().__init__(maze, neighbor_method, quiet_mode)
        self.name = "Bi Directional"

    def solve(self):

        """Function that implements a bidirectional depth-first recursive backtracker algorithm for
        solving the maze, i.e. starting at the entry point and exit points where each search searches
        for the other search path. NOTE: THE FUNCTION ENDS IN AN INFINITE LOOP FOR SOME RARE CASES OF
        THE INPUT MAZE. WILL BE FIXED IN FUTURE."""
        logging.debug("Class BiDirectional solve called")

        grid = self.maze.grid
        k_curr, l_curr = self.maze.entry_coor            # Where to start the first search
        p_curr, q_curr = self.maze.exit_coor             # Where to start the second search
        grid[k_curr][l_curr].visited = True    # Set initial cell to visited
        grid[p_curr][q_curr].visited = True    # Set final cell to visited
        backtrack_kl = list()                  # Stack of visited cells for backtracking
        backtrack_pq = list()                  # Stack of visited cells for backtracking
        path_kl = list()                       # To track path of solution and backtracking cells
        path_pq = list()                       # To track path of solution and backtracking cells

        if not self.quiet_mode:
            print("\nSolving the maze with bidirectional depth-first search...")
        time_start = time.time()

        while True:   # Loop until return statement is encountered
            neighbours_kl = self.maze.find_neighbours(k_curr, l_curr)    # Find neighbours for first search
            real_neighbours_kl = [neigh for neigh in neighbours_kl if not grid[k_curr][l_curr].is_walls_between(grid[neigh[0]][neigh[1]])]
            neighbours_kl = [neigh for neigh in real_neighbours_kl if not grid[neigh[0]][neigh[1]].visited]

            neighbours_pq = self.maze.find_neighbours(p_curr, q_curr)    # Find neighbours for second search
            real_neighbours_pq = [neigh for neigh in neighbours_pq if not grid[p_curr][q_curr].is_walls_between(grid[neigh[0]][neigh[1]])]
            neighbours_pq = [neigh for neigh in real_neighbours_pq if not grid[neigh[0]][neigh[1]].visited]

            if len(neighbours_kl) > 0:   # If there are unvisited neighbour cells
                backtrack_kl.append((k_curr, l_curr))              # Add current cell to stack
                path_kl.append(((k_curr, l_curr), False))          # Add coordinates to part of search path
                k_next, l_next = random.choice(neighbours_kl)      # Choose random neighbour
                grid[k_next][l_next].visited = True                # Move to that neighbour
                k_curr = k_next
                l_curr = l_next

            elif len(backtrack_kl) > 0:                  # If there are no unvisited neighbour cells
                path_kl.append(((k_curr, l_curr), True))   # Add coordinates to part of search path
                k_curr, l_curr = backtrack_kl.pop()        # Pop previous visited cell (backtracking)

            if len(neighbours_pq) > 0:                        # If there are unvisited neighbour cells
                backtrack_pq.append((p_curr, q_curr))           # Add current cell to stack
                path_pq.append(((p_curr, q_curr), False))       # Add coordinates to part of search path
                p_next, q_next = random.choice(neighbours_pq)   # Choose random neighbour
                grid[p_next][q_next].visited = True             # Move to that neighbour
                p_curr = p_next
                q_curr = q_next

            elif len(backtrack_pq) > 0:                  # If there are no unvisited neighbour cells
                path_pq.append(((p_curr, q_curr), True))   # Add coordinates to part of search path
                p_curr, q_curr = backtrack_pq.pop()        # Pop previous visited cell (backtracking)

            # Exit loop and return path if any opf the kl neighbours are in path_pq.
            if any((True for n_kl in real_neighbours_kl if (n_kl, False) in path_pq)):
                path_kl.append(((k_curr, l_curr), False))
                path = [p_el for p_tuple in zip(path_kl, path_pq) for p_el in p_tuple]  # Zip paths
                if not self.quiet_mode:
                    print("Number of moves performed: {}".format(len(path)))
                    print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))
                logging.debug("Class BiDirectional leaving solve")
                return path

            # Exit loop and return path if any opf the pq neighbours are in path_kl.
            elif any((True for n_pq in real_neighbours_pq if (n_pq, False) in path_kl)):
                path_pq.append(((p_curr, q_curr), False))
                path = [p_el for p_tuple in zip(path_kl, path_pq) for p_el in p_tuple]  # Zip paths
                if not self.quiet_mode:
                    print("Number of moves performed: {}".format(len(path)))
                    print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))
                logging.debug("Class BiDirectional leaving solve")
                return path


class DepthFirstBacktracker(Solver):
    """A solver that implements the depth-first recursive backtracker algorithm.
    """

    def __init__(self, maze, quiet_mode=False,  neighbor_method="fancy"):
        logging.debug('Class DepthFirstBacktracker ctor called')

        super().__init__(maze, neighbor_method, quiet_mode)
        self.name = "Depth First Backtracker"

    def solve(self):
        logging.debug("Class DepthFirstBacktracker solve called")
        k_curr, l_curr = self.maze.entry_coor      # Where to start searching
        self.maze.grid[k_curr][l_curr].visited = True     # Set initial cell to visited
        visited_cells = list()                  # Stack of visited cells for backtracking
        path = list()                           # To track path of solution and backtracking cells
        if not self.quiet_mode:
            print("\nSolving the maze with depth-first search...")

        time_start = time.time()

        while (k_curr, l_curr) != self.maze.exit_coor:     # While the exit cell has not been encountered
            neighbour_indices = self.maze.find_neighbours(k_curr, l_curr)    # Find neighbour indices
            neighbour_indices = self.maze.validate_neighbours_solve(neighbour_indices, k_curr,
                l_curr, self.maze.exit_coor[0], self.maze.exit_coor[1], self.neighbor_method)

            if neighbour_indices is not None:   # If there are unvisited neighbour cells
                visited_cells.append((k_curr, l_curr))              # Add current cell to stack
                path.append(((k_curr, l_curr), False))  # Add coordinates to part of search path
                k_next, l_next = random.choice(neighbour_indices)   # Choose random neighbour
                self.maze.grid[k_next][l_next].visited = True                 # Move to that neighbour
                k_curr = k_next
                l_curr = l_next

            elif len(visited_cells) > 0:              # If there are no unvisited neighbour cells
                path.append(((k_curr, l_curr), True))   # Add coordinates to part of search path
                k_curr, l_curr = visited_cells.pop()    # Pop previous visited cell (backtracking)

        path.append(((k_curr, l_curr), False))  # Append final location to path
        if not self.quiet_mode:
            print("Number of moves performed: {}".format(len(path)))
            print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))

        logging.debug('Class DepthFirstBacktracker leaving solve')
        return path
