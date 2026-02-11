import numpy as np
import networkx as nx
from scipy.optimize import linprog
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- CONFIGURATION ---
POPULATION_SIZE = 100
GENERATIONS = 5000  # Needs more time to find fragile structures
NUM_RECTANGLES = 20  # Try 20-25 to allow for complex cycles
CANVAS_SIZE = 1000

def get_intersection_graph(rects):
    """
    Builds the intersection graph.
    Returns: Graph G
    """
    n = len(rects)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Efficient intersection check
    # rect is [x, y, w, h]
    # x2 = x + w, y2 = y + h
    coords = np.array(rects)
    x1 = coords[:, 0]
    y1 = coords[:, 1]
    x2 = coords[:, 0] + coords[:, 2]
    y2 = coords[:, 1] + coords[:, 3]
    
    # Broadcast comparison for all pairs
    # Intersect if not (r1_x2 < r2_x1 or ...)
    # This is equivalent to max(r1_x1, r2_x1) < min(r1_x2, r2_x2) ...
    
    for i in range(n):
        for j in range(i + 1, n):
            # Standard AABB intersection
            if (x1[i] < x2[j] and x2[i] > x1[j] and 
                y1[i] < y2[j] and y2[i] > y1[j]):
                G.add_edge(i, j)
    return G

def fitness_function(rects):
    G = get_intersection_graph(rects)
    
    # --- HARD CONSTRAINT: TRIANGLE FREE ---
    # Helly's Theorem: For axis-parallel rectangles, if every pair in a set of 3 
    # intersects, then there is a common point of intersection.
    # Therefore, checking for Graph Triangles (K3) is sufficient to detect 3-way geometric overlap.
    triangles = list(nx.triangles(G).values())
    if sum(triangles) > 0:
        return 0.0, 0, 0, G # Invalid
        
    # --- CALCULATE GAP ---
    n = G.number_of_nodes()
    if n == 0: return 0.0, 0, 0, G

    # 1. ILP: Max Independent Set (Exact)
    # Equals Max Clique in Complement
    G_comp = nx.complement(G)
    clique, _ = nx.max_weight_clique(G_comp, weight=None)
    ilp_val = len(clique)
    
    if ilp_val == 0: return 0.0, 0, 0, G

    # 2. LP Relaxation
    # Minimize -sum(x) s.t. A x <= 1
    c = -np.ones(n)
    edges = list(G.edges())
    
    if not edges:
        return 1.0, n, n, G # Discrete points
        
    A = np.zeros((len(edges), n))
    b = np.ones(len(edges))
    for idx, (u, v) in enumerate(edges):
        A[idx, u] = 1
        A[idx, v] = 1
        
    bounds = [(0, 1) for _ in range(n)]
    
    # Using 'highs' for speed/stability
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.success:
        lp_val = -res.fun
        gap = lp_val / ilp_val
        return gap, lp_val, ilp_val, G
    else:
        return 0.0, 0, 0, G

def create_random_individual():
    rects = []
    for _ in range(NUM_RECTANGLES):
        x = random.randint(0, CANVAS_SIZE - 100)
        y = random.randint(0, CANVAS_SIZE - 100)
        # Bias towards "thin" rectangles to encourage cycle formation without mass overlaps
        if random.random() < 0.5:
            w = random.randint(50, 400)
            h = random.randint(10, 50)  # Horizontal bar
        else:
            w = random.randint(10, 50)  # Vertical bar
            h = random.randint(50, 400)
        rects.append([x, y, w, h])
    return rects

def mutate(rects):
    # Aggressive mutation to break triangles if they exist in parent history
    new_rects = [r[:] for r in rects]
    idx = random.randint(0, len(new_rects)-1)
    
    # Types of mutation
    mut_type = random.choice(['move', 'resize', 'flip'])
    
    if mut_type == 'move':
        new_rects[idx][0] = random.randint(0, CANVAS_SIZE - 100)
        new_rects[idx][1] = random.randint(0, CANVAS_SIZE - 100)
    elif mut_type == 'resize':
        # Retain aspect ratio logic (thinness) to maintain graph structure
        if new_rects[idx][2] > new_rects[idx][3]: # Was Horz
            new_rects[idx][3] = random.randint(10, 50) # Keep thin
            new_rects[idx][2] = random.randint(50, 400)
        else:
            new_rects[idx][2] = random.randint(10, 50)
            new_rects[idx][3] = random.randint(50, 400)
    elif mut_type == 'flip':
        # Swap w and h
        new_rects[idx][2], new_rects[idx][3] = new_rects[idx][3], new_rects[idx][2]
        
    return new_rects

# --- EVOLUTION ---
population = [create_random_individual() for _ in range(POPULATION_SIZE)]
best_valid_gap = 0
best_rects = None
best_meta = (0,0)

print(f"Searching for STRICT TRIANGLE-FREE solutions (Girth >= 4)...")
print("-" * 60)

for gen in range(GENERATIONS):
    scored_pop = []
    
    for ind in population:
        gap, lp, ilp, G = fitness_function(ind)
        # Only keep VALID solutions (gap > 0 implies triangle free due to check inside fitness)
        if gap > 0:
            scored_pop.append((gap, ind, lp, ilp))
            
    # If entire population is invalid (triangles everywhere), generate fresh randoms
    if not scored_pop:
        population = [create_random_individual() for _ in range(POPULATION_SIZE)]
        continue
        
    # Sort
    scored_pop.sort(key=lambda x: x[0], reverse=True)
    
    top_gap = scored_pop[0][0]
    
    if top_gap > best_valid_gap:
        best_valid_gap = top_gap
        best_rects = scored_pop[0][1]
        best_meta = (scored_pop[0][2], scored_pop[0][3])
        print(f"Gen {gen}: Gap {best_valid_gap:.4f} | LP={best_meta[0]:.2f} ILP={best_meta[1]} | Valid (Triangle-Free)")
        
        # 1.5 is the C5 threshold (2.5/2 = 1.25 actually). 
        # C5 is 2.5 LP / 2 ILP = 1.25. 
        # We want > 1.25. 
        # 1.5 requires specific structures.
        if best_valid_gap >= 1.5:
             print(">>> FOUND TARGET >= 1.5 <<<")
             # break # Don't break, see if we can go higher
             
    # Elitism & Breeding
    # Keep top 20%
    survivors = [x[1] for x in scored_pop[:20]]
    
    new_pop = survivors[:]
    while len(new_pop) < POPULATION_SIZE:
        parent = random.choice(survivors)
        child = mutate(parent)
        new_pop.append(child)
        
    population = new_pop

# --- VISUALIZATION ---
if best_rects:
    print(f"\nFinal Best Gap: {best_valid_gap:.4f}")
    print(f"LP: {best_meta[0]:.4f}")
    print(f"ILP: {best_meta[1]}")
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xlim(0, CANVAS_SIZE)
    ax.set_ylim(0, CANVAS_SIZE)
    ax.set_aspect('equal')
    
    # Plot Rects
    for i, r in enumerate(best_rects):
        # r = [x, y, w, h]
        rect = patches.Rectangle((r[0], r[1]), r[2], r[3], 
                                 linewidth=2, edgecolor='black', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        # Center ID
        ax.text(r[0] + r[2]/2, r[1] + r[3]/2, str(i), 
                ha='center', va='center', fontsize=9, color='blue', weight='bold')
                
    plt.title(f"Best Triangle-Free Gap: {best_valid_gap:.4f} (LP={best_meta[0]:.2f}/ILP={best_meta[1]})")
    plt.savefig("triangle_free_result.png")
    plt.show()