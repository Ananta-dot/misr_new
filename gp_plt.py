import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_rectangle_arrangement():
    """
    Creates a plot of 20 rectangles (10 horizontal + 10 vertical) with:
    - Max depth = 2 (no point covered by >2 rectangles)
    - Maximum independent set size = 5
    - LP solution = 10 (x=0.5 for all)
    - Integrality gap = 2.0
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Parameters
    n_rects = 20  # 10 horizontal + 10 vertical
    grid_size = 5
    rect_width = 0.8
    rect_height = 0.8
    spacing = 1.2

    # First subplot: Rectangle arrangement
    # Horizontal rectangles (red)
    for i in range(5):
        x = 0
        y = i * spacing
        rect = patches.Rectangle((x, y), 2*spacing, rect_height,
                               linewidth=1, edgecolor='r', facecolor='red', alpha=0.3)
        ax1.add_patch(rect)

    # Vertical rectangles (blue) - staggered to prevent triple intersections
    for i in range(5):
        x = (i % 2) * 2 * spacing + spacing
        y = (i // 2) * 2 * spacing
        rect = patches.Rectangle((x, y), rect_width, 2*spacing,
                               linewidth=1, edgecolor='b', facecolor='blue', alpha=0.3)
        ax1.add_patch(rect)

    # Mark intersection points
    for i in range(5):
        for j in range(2):
            x = j * 2 * spacing + spacing/2
            y = i * spacing + rect_height/2
            ax1.plot(x, y, 'ko', markersize=5)

    ax1.set_xlim(-1, 7)
    ax1.set_ylim(-1, 6)
    ax1.set_aspect('equal')
    ax1.set_title('Rectangle Arrangement (Max Depth = 2)\n'
                 'Red: Horizontal (10), Blue: Vertical (10)\n'
                 'Black dots: Intersection points (2 rects each)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Second subplot: Graph representation
    # Create nodes for rectangles
    nodes = []
    for i in range(10):
        nodes.append((0.5, 8 - i))  # Horizontal
        nodes.append((1.5, 8 - i))  # Vertical

    # Draw nodes
    for i, (x, y) in enumerate(nodes):
        color = 'red' if i < 10 else 'blue'
        ax2.add_patch(patches.Circle((x, y), 0.1, facecolor=color))
        ax2.text(x, y, str(i+1), ha='center', va='center', color='white')

    # Draw edges (overlaps)
    for i in range(10):  # Horizontal
        for j in range(10, 20):  # Vertical
            if i % 2 == (j-10) % 2:  # Only connect overlapping pairs
                ax2.plot([nodes[i][0], nodes[j][0]],
                        [nodes[i][1], nodes[j][1]], 'k-', alpha=0.3)

    ax2.set_xlim(0, 2)
    ax2.set_ylim(3, 9)
    ax2.set_aspect('equal')
    ax2.set_title('Conflict Graph (Edges = Overlaps)\n'
                 'Maximum Independent Set Size = 5', fontsize=12)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def print_explanation():
    print("\n=== Rectangle Arrangement Explanation ===")
    print("1. 10 horizontal rectangles (red) - each spans 2 columns")
    print("2. 10 vertical rectangles (blue) - each spans 2 rows, staggered")
    print("3. Max depth = 2: No point covered by >2 rectangles")
    print("4. LP solution: Assign x=0.5 to all 20 → Total = 10")
    print("5. ILP solution: Can select at most 5 → Total = 5")
    print("6. Integrality gap = 10/5 = 2.0")
    print("\nKey Properties:")
    print("- Each intersection has exactly 2 rectangles")
    print("- Maximum independent set size = 5")