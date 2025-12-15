import cv2
import numpy as np
import os
import glob
import sys

# --- CONFIGURATION ---
INPUT_ROOT = "processed_dataset"
OUTPUT_ROOT = "solved_puzzles_unbounded"

# TUNING (Same as before, since your local accuracy is good)
TRIM_MARGIN = 3         
PREDICTION_WEIGHT = 0.7 

class Piece:
    def __init__(self, image_path, index):
        self.index = index
        self.bgr = cv2.imread(image_path)
        self.h, self.w, _ = self.bgr.shape
        self.lab = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2LAB).astype("float32")

    def get_edge(self, orientation):
        # 0: Top, 1: Bottom, 2: Left, 3: Right
        if orientation == 0: return self.lab[0, TRIM_MARGIN:-TRIM_MARGIN]
        if orientation == 1: return self.lab[-1, TRIM_MARGIN:-TRIM_MARGIN]
        if orientation == 2: return self.lab[TRIM_MARGIN:-TRIM_MARGIN, 0]
        if orientation == 3: return self.lab[TRIM_MARGIN:-TRIM_MARGIN, -1]
        
    def get_pixel_probe(self, orientation):
        if orientation == 0: return self.lab[1, TRIM_MARGIN:-TRIM_MARGIN]
        if orientation == 1: return self.lab[-2, TRIM_MARGIN:-TRIM_MARGIN]
        if orientation == 2: return self.lab[TRIM_MARGIN:-TRIM_MARGIN, 1]
        if orientation == 3: return self.lab[TRIM_MARGIN:-TRIM_MARGIN, -2]

class Cluster:
    def __init__(self, piece_id):
        self.pieces = {piece_id: (0, 0)} # {id: (x, y)}
        # We track distinct sets for fast "is connected" checks
        self.root = self 

    def merge(self, other_cluster, relative_shift):
        dx, dy = relative_shift
        for pid, (ox, oy) in other_cluster.pieces.items():
            new_x = ox + dx
            new_y = oy + dy
            self.pieces[pid] = (new_x, new_y)
        
        # Point the other cluster to this one (Union-Find Logic)
        other_cluster.root = self

class MSTSolver:
    def __init__(self, piece_files):
        self.pieces = [Piece(f, i) for i, f in enumerate(piece_files)]
        self.num_pieces = len(self.pieces)
        self.grid_n = int(np.sqrt(self.num_pieces))
        self.clusters = [Cluster(i) for i in range(self.num_pieces)]
        self.edges = [] 

    def _calc_score(self, edge1, inner1, edge2):
        slope = edge1 - inner1
        prediction = edge1 + slope
        grad_error = np.sum(np.abs(prediction - edge2))
        ssd_error = np.sum(np.abs(edge1 - edge2))
        return (PREDICTION_WEIGHT * grad_error) + ((1 - PREDICTION_WEIGHT) * ssd_error)

    def compute_all_edges(self):
        print(f"   [Graph] Calculating edges...")
        for i in range(self.num_pieces):
            for j in range(self.num_pieces):
                if i == j: continue
                pA, pB = self.pieces[i], self.pieces[j]

                # Horizontal (A Left of B)
                cost_h = self._calc_score(pA.get_edge(3), pA.get_pixel_probe(3), pB.get_edge(2))
                self.edges.append((cost_h, i, j, 'horizontal'))

                # Vertical (A Top of B)
                cost_v = self._calc_score(pA.get_edge(1), pA.get_pixel_probe(1), pB.get_edge(0))
                self.edges.append((cost_v, i, j, 'vertical'))

        self.edges.sort(key=lambda x: x[0])

    def get_root(self, cluster):
        # Path compression for efficiency
        if cluster.root != cluster:
            cluster.root = self.get_root(cluster.root)
        return cluster.root

    def solve(self):
        self.compute_all_edges()
        
        print(f"   [Assembly] Unbounded Merging...")
        merge_count = 0
        
        for cost, p1, p2, relation in self.edges:
            c1 = self.get_root(self.clusters[p1])
            c2 = self.get_root(self.clusters[p2])
            
            if c1 is c2: continue # Already connected
            
            # Determine Shift
            p1_pos = c1.pieces[p1]
            p2_pos = c2.pieces[p2]
            
            if relation == 'horizontal': 
                target_x, target_y = p1_pos[0] + 1, p1_pos[1]
            else: 
                target_x, target_y = p1_pos[0], p1_pos[1] + 1
            
            shift_x = target_x - p2_pos[0]
            shift_y = target_y - p2_pos[1]
            
            # RELAXED CHECK: Only check for OVERLAPS, not bounds
            if self._is_valid_merge(c1, c2, shift_x, shift_y):
                c1.merge(c2, (shift_x, shift_y))
                merge_count += 1
                
                # Check if we are done
                if merge_count == self.num_pieces - 1:
                    print("   [Success] All pieces connected!")
                    break

        # FIND LARGEST CLUSTER (to handle fragmentation)
        # Group all pieces by their root cluster
        final_groups = {}
        for i in range(self.num_pieces):
            root = self.get_root(self.clusters[i])
            if root not in final_groups: final_groups[root] = []
            final_groups[root].append(i)
            
        # Pick the winner
        largest_cluster = max(final_groups, key=lambda k: len(final_groups[k]))
        print(f"   [Result] Largest Island has {len(final_groups[largest_cluster])} pieces.")
        
        return self._render_cluster(largest_cluster)

    def _is_valid_merge(self, c1, c2, dx, dy):
        # Check overlap only
        for _, (x, y) in c2.pieces.items():
            new_x = x + dx
            new_y = y + dy
            for _, (c1x, c1y) in c1.pieces.items():
                if c1x == new_x and c1y == new_y:
                    return False
        return True

    def _render_cluster(self, cluster):
        # 1. Calculate Bounds
        all_x = [p[0] for p in cluster.pieces.values()]
        all_y = [p[1] for p in cluster.pieces.values()]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        width_pieces = max_x - min_x + 1
        height_pieces = max_y - min_y + 1
        
        print(f"   [Geometry] Solution Size: {width_pieces}x{height_pieces} pieces")
        
        # 2. Create Canvas (Add margin)
        margin_px = 50
        piece_h, piece_w = self.pieces[0].h, self.pieces[0].w
        
        canvas_h = (height_pieces * piece_h) + (2 * margin_px)
        canvas_w = (width_pieces * piece_w) + (2 * margin_px)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype="uint8")
        
        # 3. Paint
        for pid, (x, y) in cluster.pieces.items():
            # Shift to 0,0 then add margin
            draw_x = ((x - min_x) * piece_w) + margin_px
            draw_y = ((y - min_y) * piece_h) + margin_px
            
            canvas[draw_y:draw_y+piece_h, draw_x:draw_x+piece_w] = self.pieces[pid].bgr
            
        return canvas

def run_unbounded_solver():
    if not os.path.exists(INPUT_ROOT): return
    puzzle_folders = []
    for root, dirs, files in os.walk(INPUT_ROOT):
        if "piece_000.png" in files: puzzle_folders.append(root)

    for p_folder in puzzle_folders:
        print(f"\n--- Solving: {os.path.basename(p_folder)} ---")
        files = sorted(glob.glob(os.path.join(p_folder, "piece_*.png")))
        if not files: continue

        solver = MSTSolver(files)
        result = solver.solve()
        
        rel_path = os.path.relpath(p_folder, INPUT_ROOT)
        save_folder = os.path.join(OUTPUT_ROOT, rel_path)
        if not os.path.exists(save_folder): os.makedirs(save_folder)
            
        cv2.imwrite(os.path.join(save_folder, "solution_unbounded.jpg"), result)
        print(f"   > Saved to: {save_folder}")

if __name__ == "__main__":
    run_unbounded_solver()