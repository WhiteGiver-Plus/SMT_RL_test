import matplotlib.pyplot as plt
import numpy as np

def draw_maze(save_path='maze.png'):
    # 创建8x8的迷宫数组
    maze = np.zeros((8, 8))
    
    # 设置墙体位置 (1表示墙)
    walls = [(0, 1), (0, 4), (5, 0), (5, 4), (1, 7), (6, 3), (6, 7), (7, 5)]
    for wall in walls:
        maze[wall] = 1
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制网格
    for i in range(9):
        ax.axhline(y=i, color='gray', linewidth=0.5)
        ax.axvline(x=i, color='gray', linewidth=0.5)
    
    # 填充墙体
    for i in range(8):
        for j in range(8):
            if maze[i, j] == 1:
                ax.fill([j, j+1, j+1, j], [8-i-1, 8-i-1, 8-i, 8-i], 'black')
    
    # 标注特殊点
    # 起点 S
    ax.text(0.3, 7.3, 'S', fontsize=15, color='blue')
    # 简单终点 E
    ax.text(2.3, 1.3, 'E', fontsize=15, color='green')
    # 困难终点 H
    ax.text(5.3, 1.3, 'H', fontsize=15, color='red')
    
    # 设置坐标轴
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    
    # 移除坐标轴标签
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 设置标题
    plt.title('Frozen Maze')
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图形
    plt.show()

if __name__ == "__main__":
    draw_maze('maze.png')  # 可以指定保存路径和文件名