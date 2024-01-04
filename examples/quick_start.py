from __future__ import absolute_import
from src.maze_manager import MazeManager
from src.maze import Maze
from os.path import join

if __name__ == "__main__":

    manager = MazeManager()

    # Create 3 random 20x20 mazes
    for x in range(1):
        maze = manager.add_maze(20, 20)
    
    # number of episodes
    episodes = 50
    params = [0.2, 0.4, 0.6, 0.8]
    test_param = "epsilon"
    hyper_tuning = False

    for maze in manager.get_mazes():
        manager.set_filename(join("img", f"Q-learn{maze.id}"))
        manager.show_maze(maze.id)
        if hyper_tuning:
            for param in params:
            # see effect of parameters
                # Solve with Q-learning
                for episode in range(episodes):
                    print(f"Maze id: {maze.id}; {test_param}: {param}; Episode #: {episode}")
                    manager.set_filename(join("img", f"Q-learn{maze.id}{test_param}{param}Episode{episode}"))
                    manager.solve_maze(maze.id, "q_learning_search", a=0.8, g=0.8, e=param, max_iter=10000)
                    if episode == 0 or episode == episodes - 1:
                        manager.show_solution_animation(maze.id)
                        manager.show_solution(maze.id)
                    maze.solution_path = None
                manager.reset(maze.id)
        else:
            for episode in range(episodes):
                print(f"Maze id: {maze.id}; Episode #: {episode}")
                manager.set_filename(join("img", f"Q-learn{maze.id}Episode{episode}"))
                manager.solve_maze(maze.id, "q_learning_search", a=0.8, g=0.9, e=0.2, max_iter=10000)
                if episode == 0 or episode == episodes - 1:
                    manager.show_solution_animation(maze.id)
                    manager.show_solution(maze.id)
                maze.solution_path = None

    
    """
    maze = manager.get_maze(0)
    for param in [0.2, 0.4, 0.6, 0.8]:
        # see effect of parameters
        manager.reset(maze.id)
        manager.solve_maze(maze.id, "q_learning_search", a=param, g=0.8, e=0.2, max_iter=10000)
        manager.set_filename(join("img", f"Q-learn{maze.id}alpha{param}"))
        manager.show_solution_animation(maze.id)
        manager.show_solution(maze.id)
    """
