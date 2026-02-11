import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# --- CONFIGURATION ---
# We need a Large N to find rare structures like the Grötzsch graph embedded in chaos.
N_START = 350
CANVAS_SIZE = 1000
ITERATIONS = 10000

def create_random_rect(id_val):
    # Generating diverse shapes: thin strips, squares, large blocks
    shape_type = random.random()
    if shape_type < 0.3:
        # Vertical Strip
        w = random.uniform(10, 50)
        h = random.uniform(200, 600)
    elif shape_type < 0.6:
        # Horizontal Strip
        w = random.uniform(200, 600)
        h = random.uniform(10, 50)
    else:
        # Chunk
        w = random.uniform(100, 300)
        h = random.uniform(100, 300)
        
    x = random.uniform(0, CANVAS_SIZE - w)
    y = random.uniform(0, CANVAS_SIZE - h)
    return {'id': id_val, 'x': x, 'y': y, 'w': w, 'h': h}

def check_intersection(r1, r2):
    return not (r1['x'] + r1['w'] <= r2['x'] or r2['x'] + r2['w'] <= r1['x'] or
                r1['y'] + r1['h'] <= r2['y'] or r2['y'] + r2['h'] <= r1['y'])

def solve_gap(G):
    if G.number_of_nodes() == 0: return 0, 0, 0
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 8)
    
    x = model.addVars(G.nodes(), vtype=GRB.BINARY)
    for u, v in G.edges():
        model.addConstr(x[u] + x[v] <= 1)
        
    model.setObjective(x.sum(), GRB.MAXIMIZE)
    model.optimize()
    
    if model.SolCount == 0: return 0,0,0
    ilp = model.objVal
    
    if ilp == 0: return 0, 0, 0

    # LP Relaxation
    for v in model.getVars(): v.vtype = GRB.CONTINUOUS
    model.optimize()
    lp = model.objVal
    
    return lp, ilp, lp/ilp

def smart_prune(G_orig):
    """
    Removes nodes involved in the MOST triangles first.
    This preserves the dense, complex structure better than random deletion.
    """
    G = G_orig.copy()
    
    while True:
        # 1. Get dictionary of triangle counts per node
        # This is expensive, so we do it iteratively
        tri_counts = nx.triangles(G)
        
        # 2. Check if done
        total_tris = sum(tri_counts.values())
        if total_tris == 0:
            break
            
        # 3. Find max offender
        # Sort nodes by triangle count (descending)
        # We introduce a little randomness to escape local optima
        candidates = sorted(tri_counts.items(), key=lambda item: item[1], reverse=True)
        
        # Pick one of the top 3 offenders to kill
        top_k = 3
        if len(candidates) < top_k: top_k = len(candidates)
        
        victim = candidates[random.randint(0, top_k-1)][0]
        
        # 4. Remove
        G.remove_node(victim)
        
    return G

# --- MAIN LOOP ---
print(f"--- Starting Dense Core Miner (N={N_START}) ---")
print("Generating massive graphs and surgically removing triangles...")

best_gap = 1.0
best_rects = []
best_G = None

for i in range(1, ITERATIONS + 1):
    # 1. Generate Massive Chaos
    rects = [create_random_rect(k) for k in range(N_START)]
    
    # 2. Build Graph (Optimized)
    G_full = nx.Graph()
    G_full.add_nodes_from(range(N_START))
    
    # Fast Edge Check
    for idx1 in range(N_START):
        for idx2 in range(idx1 + 1, N_START):
            if check_intersection(rects[idx1], rects[idx2]):
                G_full.add_edge(idx1, idx2)
                
    # 3. Smart Prune
    # print(f"Run {i}: Pruning {G_full.number_of_nodes()} nodes...", end="")
    G_clean = smart_prune(G_full)
    # print(f" -> {G_clean.number_of_nodes()} nodes left. Solving...", end="")
    
    # 4. Check Gap
    if G_clean.number_of_nodes() > 5:
        lp, ilp, gap = solve_gap(G_clean)
        
        print(f"Run {i}: Nodes={G_clean.number_of_nodes()} Gap={gap:.4f} (LP={lp:.2f}/ILP={ilp:.0f})")
        
        if gap > best_gap:
            best_gap = gap
            best_G = G_clean
            best_rects = [r for r in rects if r['id'] in G_clean.nodes()]
            print(f"  >>> NEW BEST GAP: {best_gap:.4f}")
            
            if best_gap >= 1.25:
                print("  !!! HIT THE PENTAGON BARRIER OR HIGHER !!!")
                
                # Save immediately
                plt.figure(figsize=(10,10))
                ax = plt.gca()
                ax.set_xlim(0, CANVAS_SIZE); ax.set_ylim(0, CANVAS_SIZE)
                for r in best_rects:
                    c = 'red' if gap > 1.2 else 'blue'
                    ax.add_patch(patches.Rectangle((r['x'], r['y']), r['w'], r['h'], 
                                                   facecolor=c, edgecolor='black', alpha=0.5))
                plt.title(f"High Gap Structure (Gap={best_gap:.4f})")
                plt.savefig(f"dense_miner_{best_gap:.4f}.png")
                
    else:
        print(f"Run {i}: Graph died (too sparse)")

print("="*40)
print(f"Final Best Gap: {best_gap:.4f}")