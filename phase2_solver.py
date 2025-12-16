import cv2
import numpy as np
import os
import glob
import sys
import math

# --- CONFIGURATION ---
INPUT_ROOT = "phase 1 processed" 
OUTPUT_ROOT = "phase 2 solutions"
SHOW_PROGRESS = True

# TUNING FOR 8x8
TRIM_MARGIN = 2          # Increased slightly to avoid corner noise
PREDICTION_WEIGHT = 0.75 # Higher weight on gradient for better flow
RATIO_THRESHOLD = 0.85   
BEST_BUDDY_BONUS = 1e9

class Piece:
    def __init__(self, image_path, index):
        self.index = index
        self.bgr = cv2.imread(image_path)
        self.h, self.w, _ = self.bgr.shape
        self.lab = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2LAB).astype("float32")

    def get_rows(self, row_index):
        data = self.lab[row_index, :, :]
        return data[TRIM_MARGIN : -TRIM_MARGIN, :]

    def get_cols(self, col_index):
        data = self.lab[:, col_index, :]
        return data[TRIM_MARGIN : -TRIM_MARGIN, :]

class Solver:
    def __init__(self, piece_files):
        self.pieces = [Piece(f, i) for i, f in enumerate(piece_files)]
        self.num_pieces = len(self.pieces)
        self.grid_n = int(np.sqrt(self.num_pieces))
        
        self.cost_matrix = np.zeros((self.num_pieces, self.num_pieces, 2), dtype=np.float32)
        self.bb_horizontal = {} 
        self.bb_vertical = {}
        
        # Track connectivity for smart seeding
        self.connectivity_score = np.zeros(self.num_pieces, dtype=int)

    def _compute_mixed_error(self, trend_1, trend_2, target):
        """
        ENHANCEMENT 1: Variance Weighted Error
        Normalizes the error by the texture complexity. 
        Allows high-contrast matches (text) to be scored fairly against low-contrast matches (sky).
        """
        slope = trend_1 - trend_2
        prediction = trend_1 + slope
        
        # Calculate raw errors
        error_pred = np.sum(np.abs(prediction - target))
        error_ssd = np.sum(np.abs(trend_1 - target))
        
        raw_score = (PREDICTION_WEIGHT * error_pred) + ((1 - PREDICTION_WEIGHT) * error_ssd)
        
        # Calculate Variance (Texture Complexity) of the junction
        # We add epsilon (1.0) to prevent division by zero in flat areas
        variance = np.std(trend_1) + np.std(target) + 1.0
        
        # Normalize: Penalize errors less in high-variance areas
        return raw_score / variance

    def compute_compatibility_matrix(self):
        if SHOW_PROGRESS: print(f"   [Analysis] Pre-computing normalized costs...")
        
        for i, pA in enumerate(self.pieces):
            for j, pB in enumerate(self.pieces):
                if i == j:
                    self.cost_matrix[i, j, :] = float('inf')
                    continue

                # Horizontal (A Left of B)
                col_last = pA.get_cols(pA.w - 1)
                col_prev = pA.get_cols(pA.w - 2)
                col_target = pB.get_cols(0)
                self.cost_matrix[i, j, 0] = self._compute_mixed_error(col_last, col_prev, col_target)

                # Vertical (A Top of B)
                row_last = pA.get_rows(pA.h - 1)
                row_prev = pA.get_rows(pA.h - 2)
                row_target = pB.get_rows(0)
                self.cost_matrix[i, j, 1] = self._compute_mixed_error(row_last, row_prev, row_target)

    def find_robust_best_buddies(self):
        def get_best(costs):
            indices = np.argsort(costs)
            best, second = indices[0], indices[1]
            ratio = costs[best] / (costs[second] + 1e-5)
            return best, ratio

        # Horizontal Analysis
        best_right = {}
        best_left = {}
        for i in range(self.num_pieces):
            j, r = get_best(self.cost_matrix[i, :, 0])
            if r < RATIO_THRESHOLD: best_right[i] = j
        for j in range(self.num_pieces):
            i, r = get_best(self.cost_matrix[:, j, 0])
            if r < RATIO_THRESHOLD: best_left[j] = i
            
        for i, j in best_right.items():
            if best_left.get(j) == i: 
                self.bb_horizontal[i] = j
                # ENHANCEMENT 2: Track Connectivity
                self.connectivity_score[i] += 1
                self.connectivity_score[j] += 1

        # Vertical Analysis
        best_bottom = {}
        best_top = {}
        for i in range(self.num_pieces):
            j, r = get_best(self.cost_matrix[i, :, 1])
            if r < RATIO_THRESHOLD: best_bottom[i] = j
        for j in range(self.num_pieces):
            i, r = get_best(self.cost_matrix[:, j, 1])
            if r < RATIO_THRESHOLD: best_top[j] = i
            
        for i, j in best_bottom.items():
            if best_top.get(j) == i: 
                self.bb_vertical[i] = j
                self.connectivity_score[i] += 1
                self.connectivity_score[j] += 1

        if SHOW_PROGRESS:
            print(f"   [Analysis] Robust Buddies: {len(self.bb_horizontal)} H / {len(self.bb_vertical)} V")

    def solve(self):
        self.compute_compatibility_matrix()
        self.find_robust_best_buddies()
        
        best_grid = None
        min_error = float('inf')

        # ENHANCEMENT 3: Smart Seed Ordering
        # Sort pieces by connectivity. Start with the pieces that have the most "Best Buddies".
        # This dramatically increases the chance of finding the correct solution early.
        sorted_seeds = np.argsort(self.connectivity_score)[::-1] # Descending order

        print(f"   [Assembly] Testing seeds (Prioritizing highly connected pieces)...")
        
        # Only try the top 50% of connected seeds to save time, or all if dataset is small
        # For 8x8 (64 pieces), checking all 64 is fine, but sorting helps us find the min_error faster.
        for i, start_node in enumerate(sorted_seeds):
            if SHOW_PROGRESS: sys.stdout.write("."); sys.stdout.flush()
            
            grid, score = self._assemble(start_node)
            
            if score < min_error:
                min_error = score
                best_grid = grid

        print(f"\n   [Result] Best Score: {min_error:.2f}")
        return self._render_result(best_grid)

    def _assemble(self, start_node):
        grid = np.full((self.grid_n, self.grid_n), -1, dtype=int)
        used = {start_node}
        grid[0, 0] = start_node
        
        frontier = set()
        frontier.add((0, 1))
        frontier.add((1, 0))
        
        total_score = 0
        
        for _ in range(self.num_pieces - 1):
            if not frontier: break
            
            # Find Most Constrained Slot
            best_slot = None
            max_neighbors = -1
            sorted_frontier = []
            
            for (r, c) in frontier:
                neighbors = 0
                if r > 0 and grid[r-1, c] != -1: neighbors += 1
                if r < self.grid_n - 1 and grid[r+1, c] != -1: neighbors += 1
                if c > 0 and grid[r, c-1] != -1: neighbors += 1
                if c < self.grid_n - 1 and grid[r, c+1] != -1: neighbors += 1
                sorted_frontier.append( (neighbors, r, c) )
            
            sorted_frontier.sort(key=lambda x: x[0], reverse=True)
            _, target_r, target_c = sorted_frontier[0]
            
            # Find Best Piece
            best_cand = -1
            best_score = float('inf')
            
            top_idx = grid[target_r-1, target_c] if target_r > 0 else -1
            bot_idx = grid[target_r+1, target_c] if target_r < self.grid_n - 1 else -1
            left_idx = grid[target_r, target_c-1] if target_c > 0 else -1
            right_idx = grid[target_r, target_c+1] if target_c < self.grid_n - 1 else -1
            
            for cand in range(self.num_pieces):
                if cand in used: continue
                
                score = 0
                count = 0
                is_buddy = False
                
                # Check Neighbors
                if top_idx != -1:
                    c = self.cost_matrix[top_idx, cand, 1]
                    if self.bb_vertical.get(top_idx) == cand:
                        c -= BEST_BUDDY_BONUS
                        is_buddy = True
                    score += c
                    count += 1
                    
                if bot_idx != -1:
                    c = self.cost_matrix[cand, bot_idx, 1]
                    if self.bb_vertical.get(cand) == bot_idx:
                        c -= BEST_BUDDY_BONUS
                        is_buddy = True
                    score += c
                    count += 1
                
                if left_idx != -1:
                    c = self.cost_matrix[left_idx, cand, 0]
                    if self.bb_horizontal.get(left_idx) == cand:
                        c -= BEST_BUDDY_BONUS
                        is_buddy = True
                    score += c
                    count += 1
                
                if right_idx != -1:
                    c = self.cost_matrix[cand, right_idx, 0]
                    if self.bb_horizontal.get(cand) == right_idx:
                        c -= BEST_BUDDY_BONUS
                        is_buddy = True
                    score += c
                    count += 1
                    
                if count > 0:
                    avg = score / count if not is_buddy else score
                    if avg < best_score:
                        best_score = avg
                        best_cand = cand
            
            if best_cand != -1:
                grid[target_r, target_c] = best_cand
                used.add(best_cand)
                total_score += best_score
                
                frontier.remove((target_r, target_c))
                nbs = [(target_r-1, target_c), (target_r+1, target_c), 
                       (target_r, target_c-1), (target_r, target_c+1)]
                for (nr, nc) in nbs:
                    if 0 <= nr < self.grid_n and 0 <= nc < self.grid_n:
                        if grid[nr, nc] == -1:
                            frontier.add((nr, nc))
            else:
                frontier.remove((target_r, target_c))
                total_score += 1e9

        return grid, total_score

    def _render_result(self, grid):
        canvas_h = self.grid_n * self.pieces[0].h
        canvas_w = self.grid_n * self.pieces[0].w
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype="uint8")
        
        for r in range(self.grid_n):
            for c in range(self.grid_n):
                idx = grid[r, c]
                if idx != -1:
                    y, x = r * self.pieces[0].h, c * self.pieces[0].w
                    canvas[y:y+self.pieces[0].h, x:x+self.pieces[0].w] = self.pieces[idx].bgr
        return canvas

def run_solver():
    if not os.path.exists(INPUT_ROOT): return
    
    puzzle_folders = []
    for root, dirs, files in os.walk(INPUT_ROOT):
        if "piece_000.png" in files:
            puzzle_folders.append(root)

    for p_folder in puzzle_folders:
        print(f"\n--- Solving: {os.path.basename(p_folder)} ---")
        files = sorted(glob.glob(os.path.join(p_folder, "piece_*.png")))
        if not files: continue

        solver = Solver(files)
        result = solver.solve()
        
        parts = p_folder.replace("\\", "/").split("/")
        puzzle_name = os.path.basename(p_folder)
        puzzle_num = puzzle_name.split("_")[-1]
        
        category = None
        for part in parts:
            if part.startswith("puzzle_") and "x" in part:
                category = part
                break
        
        if category:
            save_folder = os.path.join(OUTPUT_ROOT, category)
            if not os.path.exists(save_folder): os.makedirs(save_folder)
            
            output_path = os.path.join(save_folder, f"{puzzle_num}.jpg")
            cv2.imwrite(output_path, result)
            print(f"   > Saved to: {output_path}")

if __name__ == "__main__":
    run_solver()