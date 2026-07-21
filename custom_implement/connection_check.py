import numpy as np
import copy

def dfs_iterative(grid, visited, start_row, start_col):
    stack = [(start_row, start_col)]
    visited[start_row, start_col] = 1 # 현재 위치를 방문했음을 표시합니다.

    while stack:
        row, col = stack.pop()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]: # 이동할 수 있는 방향: 상, 하, 좌, 우
            nr, nc = row + dr, col + dc
            if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and
                grid[nr, nc] == 0 and not visited[nr, nc]):
                visited[nr, nc] = 1
                stack.append((nr, nc))

def is_connected(grid):
    """
    주어진 그리드에서 모든 칸이 연결되어 있는지 확인합니다.
    """
    # 방문 여부를 나타내는 배열을 생성합니다. 초기값은 모두 False로 설정합니다.
    visited = copy.deepcopy(grid)

    # 시작 지점을 찾습니다. 장애물이 없는 첫 번째 칸을 시작점으로 선택합니다.
    start_row, start_col = np.where(grid == 0)
    start_row, start_col = start_row[0], start_col[0]

    # DFS를 사용하여 그리드에서 모든 칸이 연결되어 있는지 확인합니다.
    dfs_iterative(grid, visited, start_row, start_col)

    # 모든 칸을 방문했는지 확인합니다.
    return np.all(visited==1)