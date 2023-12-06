import heapq
import random
from collections import deque
import pygame
import json
import time
import pandas as pd

WIDTH, HEIGHT = 800, 600
ROWS, COLS = 21, 20
CELL_WIDTH = WIDTH // COLS
CELL_HEIGHT = HEIGHT // ROWS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
PINK = (255, 192, 203)

# def add_extra_obstacles(x, y, width, height, maze):
#     # Add extra obstacles or blocks within the maze in a specific pattern
#     for i in range(x + 2, x + width - 1, 3):
#         for j in range(y + 2, y + height - 1, 3):
#             maze[j][i] = {'type': '%', 'cost': -1}  # Place obstacles at intervals (adjust as needed)

def generate_maze(loop, rows, cols):
    maze = [[{'type': '%', 'cost': -1} for _ in range(cols)] for _ in range(rows)]

    # Set the start point
    start = (0, 0)
    maze[start[0]][start[1]] = {'type': ' ', 'cost': 0}  # Set start point as empty space with cost 0

    # Adjust the recursive division function to create multiple paths with walls
    def recursive_division_with_path(x, y, width, height, maze, start, end):
        if width < 3 or height < 3:
            return

        # Randomly determine the number of paths to create
        num_paths = random.randint(2, 4)  # Adjust the range as needed

        for _ in range(num_paths):
            horizontal = random.choice([True, False])

            if horizontal:
                passage_y = random.randint(y + 1, y + height - 2)
                wall_x = random.randint(x, x + width - 1)
                for i in range(x, x + width):
                    if i != wall_x:
                        maze[passage_y][i] = {'type': ' ', 'cost': random.randint(1, 5)}

                # Create wall on the path
                wall_y = random.choice([y, y + height - 1])
                maze[wall_y][wall_x] = {'type': '%', 'cost': -1}

                recursive_division_with_path(x, y, width, passage_y - y, maze, start, end)
                recursive_division_with_path(x, passage_y + 1, width, y + height - passage_y - 1, maze, start, end)
            else:
                passage_x = random.randint(x + 1, x + width - 2)
                wall_y = random.randint(y, y + height - 1)
                for i in range(y, y + height):
                    if i != wall_y:
                        maze[i][passage_x] = {'type': ' ', 'cost': random.randint(1, 5)}

                # Create wall on the path
                wall_x = random.choice([x, x + width - 1])
                maze[wall_y][wall_x] = {'type': '%', 'cost': -1}

                recursive_division_with_path(x, y, passage_x - x, height, maze, start, end)
                recursive_division_with_path(passage_x + 1, y, x + width - passage_x - 1, height, maze, start, end)

        # Set the start and end points as empty space with cost 0
        maze[start[0]][start[1]] = {'type': ' ', 'cost': 0}
        maze[end[0]][end[1]] = {'type': ' ', 'cost': 0}

        # Ensure a path from start to end
        current = start
        while current != end:
            next_move = (end[0] - current[0], end[1] - current[1])
            # Avoid division by zero by checking for zero values
            next_position = (current[0] + next_move[0] // abs(next_move[0] or 1), current[1] + next_move[1] // abs(next_move[1] or 1))
            maze[next_position[0]][next_position[1]] = {'type': ' ', 'cost': 0}
            current = next_position

    # Randomly select the endpoint
    end_x = random.randint(0, cols - 1)
    end_y = random.randint(0, rows - 1)
    endpoint = (end_y, end_x)
    maze[endpoint[0]][endpoint[1]] = {'type': ' ', 'cost': 0}
    
    def add_extra_obstacles(x, y, width, height, maze):
        # Add extra obstacles or blocks within the maze in a specific pattern
        for i in range(x + 2, x + width - 1, 3):
            for j in range(y + 2, y + height - 1, 3):
                maze[j][i] = {'type': '%', 'cost': -1}  # Place obstacles at intervals (adjust as needed)

    # Call the function to add extra obstacles to the maze
    add_extra_obstacles(0, 0, cols, rows, maze)

    recursive_division_with_path(0, 0, cols, rows, maze, start, endpoint)


    # Save maze to a file if needed
    file_name = 'mazes/maze_' + str(loop + 1) + '.json'
    with open(file_name, 'w') as file:
        json.dump(maze, file)

    return maze, endpoint



# def generate_maze(loop, rows, cols):
#     maze = [[{'type': '%', 'cost': -1} for _ in range(cols)] for _ in range(rows)]

#     # Set the start point
#     start = (0, 0)
#     maze[start[0]][start[1]] = ' '

#     # Adjust the recursive division function to create multiple paths with walls
#     def recursive_division_with_path(x, y, width, height, maze, start, end):
#         if width < 3 or height < 3:
#             return

#         # Randomly determine the number of paths to create
#         num_paths = random.randint(2, 4)  # Adjust the range as needed

#         for _ in range(num_paths):
#             horizontal = random.choice([True, False])

#             if horizontal:
#                 passage_y = random.randint(y + 1, y + height - 2)
#                 wall_x = random.randint(x, x + width - 1)
#                 for i in range(x, x + width):
#                     if i != wall_x:
#                         maze[passage_y][i] = {'type': ' ', 'cost': random.randint(1, 5)}  # Assign a random cost for path cells

#                 # Create wall on the path
#                 wall_y = random.choice([y, y + height - 1])
#                 maze[wall_y][wall_x] = '%'

#                 recursive_division_with_path(x, y, width, passage_y - y, maze, start, end)
#                 recursive_division_with_path(x, passage_y + 1, width, y + height - passage_y - 1, maze, start, end)
#             else:
#                 passage_x = random.randint(x + 1, x + width - 2)
#                 wall_y = random.randint(y, y + height - 1)
#                 for i in range(y, y + height):
#                     if i != wall_y:
#                         maze[i][passage_x] = {'type': ' ', 'cost': random.randint(1, 5)}  # Assign a random cost for path cells

#                 # Create wall on the path
#                 wall_x = random.choice([x, x + width - 1])
#                 maze[wall_y][wall_x] = '%'

#                 recursive_division_with_path(x, y, passage_x - x, height, maze, start, end)
#                 recursive_division_with_path(passage_x + 1, y, x + width - passage_x - 1, height, maze, start, end)
                
#         for row in range(rows):
#             for col in range(cols):
#                 if maze[row][col]['type'] == ' ':
#                     maze[row][col]['cost'] = random.randint(1, 5)  # Assign random costs for path cells

                
#         def add_extra_obstacles(x, y, width, height, maze):
#         # Add extra obstacles or blocks within the maze in a specific pattern
#             for i in range(x + 2, x + width - 1, 3):
#                 for j in range(y + 2, y + height - 1, 3):
#                     maze[j][i] = '%'  # Place obstacles at intervals (adjust as needed)
        
#         add_extra_obstacles(x, y, width, height, maze)

#         maze[start[0]][start[1]] = ' '  # Set start point as empty space
#         maze[end[0]][end[1]] = ' '      # Set end point as empty space

#         # Ensure a path from start to end
#         current = start
#         while current != end:
#             next_move = (end[0] - current[0], end[1] - current[1])
#             # Avoid division by zero by checking for zero values
#             next_position = (current[0] + next_move[0] // abs(next_move[0] or 1), current[1] + next_move[1] // abs(next_move[1] or 1))
#             maze[next_position[0]][next_position[1]] = ' '
#             current = next_position
    
#     # Randomly select the endpoint
#     end_x = random.randint(0, cols - 1)
#     end_y = random.randint(0, rows - 1)
#     endpoint = (end_y, end_x)
#     maze[endpoint[0]][endpoint[1]] = ' '
    
#     recursive_division_with_path(0, 0, cols, rows, maze, start, endpoint)

#     # Save maze to a file if needed
#     file_name = 'mazes/maze_' + str(loop + 1) + '.json'
#     with open(file_name, 'w') as file:
#         json.dump(maze, file)

#     return maze, endpoint


def bfs(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    queue = deque([(start, [])])
    nodes_visited = 0  # Track the number of nodes visited

    visited[start[0]][start[1]] = True  # Mark the start node as visited

    while queue:
        (r, c), path = queue.popleft()
        nodes_visited += 1  # Increment the count of visited nodes

        if (r, c) == end:
            return path + [(r, c)], nodes_visited  # Return path and nodes visited

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc]['type'] == ' ' and not visited[nr][nc]:
                visited[nr][nc] = True  # Mark the node as visited before enqueueing
                queue.append(((nr, nc), path + [(r, c)]))

    return [], nodes_visited  # If no path found, return an empty path and nodes visited

def dfs(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    stack = [(start, [])]
    nodes_visited = 0  # Track the number of nodes visited

    while stack:
        (r, c), path = stack.pop()
        nodes_visited += 1  # Increment the count of visited nodes
        if (r, c) == end:
            return path + [(r, c)], nodes_visited  # Return path and nodes visited
        if visited[r][c]:
            continue
        visited[r][c] = True
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc]['type'] == ' ' and not visited[nr][nc]:
                stack.append(((nr, nc), path + [(r, c)]))

    return [], nodes_visited


# def ucs(maze, start, end):
#     rows, cols = len(maze), len(maze[0])
#     visited = [[False for _ in range(cols)] for _ in range(rows)]
#     queue = [(0, start, [])]  # Priority queue with (cost, node, path)
#     nodes_visited = 0

#     while queue:
#         cost, (r, c), path = heapq.heappop(queue)
#         nodes_visited += 1

#         if (r, c) == end:
#             return path + [(r, c)], nodes_visited

#         if visited[r][c]:
#             continue
#         visited[r][c] = True

#         for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#             nr, nc = r + dr, c + dc
#             if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == ' ' and not visited[nr][nc]:
#                 new_cost = len(path) + 1  # Increment the cost by 1 for each step from start to current node
#                 heapq.heappush(queue, (new_cost, (nr, nc), path + [(r, c)]))

#     return [], nodes_visited

def ucs(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    queue = [(0, start, [])]  # Priority queue with (cost, node, path)
    nodes_visited = 0
    total_cost = 0  # Initialize total_cost
    
    while queue:
        cost, (r, c), path = heapq.heappop(queue)
        nodes_visited += 1

        if (r, c) == end:
            return path + [(r, c)], total_cost, nodes_visited  # Return path, total_cost, and nodes_visited

        if visited[r][c]:
            continue
        visited[r][c] = True

        # Check if the cell is open (' ') and not yet visited
        if maze[r][c]['type'] == ' ':
            total_cost = cost  # Update total_cost with the current cost

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc]['type'] == ' ' and not visited[nr][nc]:
                    new_cost = cost + maze[nr][nc]['cost']  # Add cell cost to the cumulative cost
                    heapq.heappush(queue, (new_cost, (nr, nc), path + [(r, c)]))

    return [], total_cost, nodes_visited


def draw_maze(screen, maze, endpoint):
    for r in range(len(maze)):
        for c in range(len(maze[0])):
            if maze[r][c]['type'] == '%':
                color = BLACK
            else:
                color = WHITE
            pygame.draw.rect(screen, color, (c * CELL_WIDTH, r * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))
            if (r, c) == (0, 0):  # Start point
                pygame.draw.rect(screen, RED, (c * CELL_WIDTH, r * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))
            if (r, c) == endpoint:  # End point
                pygame.draw.rect(screen, BLUE, (c * CELL_WIDTH, r * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))

def draw_path(screen, path, color):
    for r, c in path:
        pygame.draw.rect(screen, color, (c * CELL_WIDTH, r * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))


def draw_explored_nodes(screen, maze, explored_nodes):
    for r, c in explored_nodes:
        pygame.draw.rect(screen, GREEN, (c * CELL_WIDTH, r * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))

def draw_current_node(screen, maze, current_node):
    r, c = current_node
    pygame.draw.rect(screen, (0, 0, 255), (c * CELL_WIDTH, r * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))


def visualize_algorithm(loop, maze, algorithm, endpoint):
    if algorithm == "bfs":
        path, nodes_visited = bfs(maze, (0, 0), endpoint)
        color = GREEN
        length = len(path)
    elif algorithm == "dfs":
        path, nodes_visited = dfs(maze, (0, 0), endpoint)
        color = RED
        length = len(path)
    elif algorithm == "ucs":
        path, total_cost, nodes_visited = ucs(maze, (0, 0), endpoint)
        length = total_cost
        color = PINK
    else:
        print("Invalid algorithm choice!")
        return
    
    start_time = time.time()
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(str(algorithm).upper() + " Maze Visualization")

    running = True
    current_step = 0
    explored_nodes = []

    while running:
        # Event handling for quitting the visualization
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  
                    running = False

        # Pygame screen updates
        screen.fill(BLACK)
        draw_maze(screen, maze,endpoint)

        if path:
            partial_path = path[:current_step + 1]
            draw_path(screen, partial_path, color=color)

        if current_step < len(path):
            current_node = path[current_step]
            if current_node not in explored_nodes:
                explored_nodes.append(current_node)
            # Get neighbors of current node
            neighbors = [(current_node[0] + dr, current_node[1] + dc) for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]]
            valid_neighbors = [(r, c) for r, c in neighbors if 0 <= r < ROWS and 0 <= c < COLS and maze[r][c]['type'] == ' ']
            
            # Highlight neighbors considered
            for neighbor in valid_neighbors:
                if neighbor not in explored_nodes:
                    pygame.draw.rect(screen, color, (neighbor[1] * CELL_WIDTH, neighbor[0] * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))

                # Mark current node being explored
                draw_current_node(screen, maze, current_node)
                # # Draw explored nodes
                # draw_explored_nodes(screen, maze, explored_nodes)
                

        pygame.display.flip()
        pygame.time.wait(150)  

        current_step += 1
        if current_step >= len(path):
            running = False
    
    pygame.quit()
    end_time = time.time()
    duration = end_time - start_time

    trail = {}
    trail["Trial No"] = loop+1
    trail["Algorithm"] = algorithm
    trail["Nodes Visited"] = nodes_visited
    trail["Length"] = length
    trail["Time Taken"] = duration
    return trail

def main():
    analysis = []
    algorithm = ['bfs', 'dfs', 'ucs']
    for loop in range(60):
        maze, endpoint = generate_maze(loop, ROWS, COLS)
        for algorithm_choice in algorithm:
            result = visualize_algorithm(loop, maze, algorithm_choice, endpoint)
            analysis.append(result)

    df = pd.DataFrame(analysis)

    # Save the DataFrame to an Excel file
    df.to_excel('results.xlsx', index=False)


main()
