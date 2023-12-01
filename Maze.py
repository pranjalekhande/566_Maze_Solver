import heapq
import random
from collections import deque
import pygame
import json
import time

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


def generate_maze(rows, cols):
    maze = [['%' for _ in range(cols)] for _ in range(rows)]

    def recursive_division(x, y, width, height):
        if width < 3 or height < 3:
            return

        horizontal = random.choice([True, False])

        if horizontal:
            wall_y = random.randint(y + 1, y + height - 2)
            passage_x = random.randint(x + 1, x + width - 2)

            for i in range(x, x + width):
                if i != passage_x:
                    maze[wall_y][i] = ' '
            
            recursive_division(x, y, width, wall_y - y)
            recursive_division(x, wall_y + 1, width, y + height - wall_y - 1)
        else:
            wall_x = random.randint(x + 1, x + width - 2)
            passage_y = random.randint(y + 1, y + height - 2)

            for i in range(y, y + height):
                if i != passage_y:
                    maze[i][wall_x] = ' '
            
            recursive_division(x, y, wall_x - x, height)
            recursive_division(wall_x + 1, y, x + width - wall_x - 1, height)

    recursive_division(0, 0, cols, rows)
    
    x, y = 0, 0
    while x < cols - 1 or y < rows - 1:
        maze[y][x] = ' '
        if random.random() < 0.5 and x < cols - 1:
            x += 1
        elif y < rows - 1:
            y += 1

    # Ensure a clear path from the start to the end
    maze[rows - 1][cols - 1] = ' '
    
    with open('maze.json', 'w') as file:
        json.dump(maze, file)
    
    return maze

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
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == ' ' and not visited[nr][nc]:
                visited[nr][nc] = True  # Mark the node as visited before enqueueing
                queue.append(((nr, nc), path + [(r, c)]))

    return [], nodes_visited  # If no path found, return an empty path and nodes visited


def dfs(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    visited = [[False for i in range(cols)] for j in range(rows)]
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
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == ' ' and not visited[nr][nc]:
                stack.append(((nr, nc), path + [(r, c)]))

    return [], nodes_visited

def ucs(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    queue = [(0, start, [])]  # Priority queue with (cost, node, path)
    nodes_visited = 0

    while queue:
        cost, (r, c), path = heapq.heappop(queue)
        nodes_visited += 1

        if (r, c) == end:
            return path + [(r, c)], nodes_visited

        if visited[r][c]:
            continue
        visited[r][c] = True

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == ' ' and not visited[nr][nc]:
                new_cost = len(path) + 1  # Increment the cost by 1 for each step from start to current node
                heapq.heappush(queue, (new_cost, (nr, nc), path + [(r, c)]))

    return [], nodes_visited


def draw_maze(screen, maze):
    for r in range(len(maze)):
        for c in range(len(maze[0])):
            if maze[r][c] == '%':
                color = BLACK
            else:
                color = WHITE
            pygame.draw.rect(screen, color, (c * CELL_WIDTH, r * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))
            if (r, c) == (0, 0):  # Start point
                pygame.draw.rect(screen, (255, 0, 0), (c * CELL_WIDTH, r * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))
            if (r, c) == (ROWS - 1, COLS - 1):  # End point
                pygame.draw.rect(screen, (0, 0, 255), ((COLS - 1) * CELL_WIDTH, (ROWS - 1) * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))
                
def draw_path(screen, path, color):
    for r, c in path:
        pygame.draw.rect(screen, color, (c * CELL_WIDTH, r * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))

def draw_explored_nodes(screen, maze, explored_nodes):
    for r, c in explored_nodes:
        pygame.draw.rect(screen, GREEN, (c * CELL_WIDTH, r * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))

def draw_current_node(screen, maze, current_node):
    r, c = current_node
    pygame.draw.rect(screen, (0, 0, 255), (c * CELL_WIDTH, r * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))
    
def visualize_algorithm(maze, algorithm):
    if algorithm == "bfs":
        path, nodes_visited = bfs(maze, (0, 0), (ROWS - 1, COLS - 1))
        color = GREEN
    elif algorithm == "dfs":
        path, nodes_visited = dfs(maze, (0, 0), (ROWS - 1, COLS - 1))
        color = RED
    elif algorithm == "ucs":
        path, nodes_visited = ucs(maze, (0, 0), (ROWS - 1, COLS - 1))
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

    # start_time = time.time() (should start be from here or from the one before ...)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  
                    running = False

        screen.fill(BLACK)
        draw_maze(screen, maze)

        if path:
            partial_path = path[:current_step + 1]
            draw_path(screen, partial_path, color=color)

        if current_step < len(path):
            current_node = path[current_step]
            if current_node not in explored_nodes:
                explored_nodes.append(current_node)

                # Get neighbors of current node
                neighbors = [(current_node[0] + dr, current_node[1] + dc) for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]]
                valid_neighbors = [(r, c) for r, c in neighbors if 0 <= r < ROWS and 0 <= c < COLS and maze[r][c] == ' ']

                # Highlight neighbors considered
                for neighbor in valid_neighbors:
                    if neighbor not in explored_nodes:
                        pygame.draw.rect(screen, color, (neighbor[1] * CELL_WIDTH, neighbor[0] * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))

                # Mark current node being explored
                draw_current_node(screen, maze, current_node)

        pygame.display.flip()
        pygame.time.wait(300)  

        current_step += 1
        if current_step >= len(path):
            # current_step = 0
            running = False
    
    pygame.quit()
    end_time = time.time()
    duration = end_time - start_time

    print(str(algorithm).upper(), "Nodes Visited: ", nodes_visited)
    print(str(algorithm).upper(), "Length: ", len(path))
    print(str(algorithm).upper(), "Time Taken: ", duration)

def main():
    # maze = generate_maze(ROWS, COLS)
    with open('maze.json', 'r') as file:
        maze = json.load(file)

    algorithm_choice = input("Enter 'bfs' or 'dfs' or 'ucs' to visualize the algorithm: ").lower()

    visualize_algorithm(maze, algorithm_choice)

main()
