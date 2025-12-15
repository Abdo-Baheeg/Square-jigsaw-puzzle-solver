import cv2
import numpy as np
import os
import glob
import sys
import heapq

# --- CONFIGURATION ---
INPUT_ROOT = "processed_dataset"
OUTPUT_ROOT = "solved_puzzles"
SHOW_PROGRESS = True

# TUNING FOR 8x8
TRIM_MARGIN = 1          
PREDICTION_WEIGHT = 0.70 
RATIO_THRESHOLD = 0.85   # Strictness for Best Buddies
BEST_BUDDY_BONUS = 1e9

class Piece:
    def __init__(self, image_path, index):
        self.index = index
        self.bgr = cv2.imread(image_path)
        self.h, self.w, _ = self.bgr.shape
        self.lab = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2LAB).astype("float32")

    def get_rows(self, row_index):
        # 2D Slicing Fix
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
        
        # [Piece_A][Piece_B][0=Horiz, 1=Vert]
        self.cost_matrix = np.zeros((self.num_pieces, self.num_pieces, 2), dtype=np.float32)
        
        self.bb_horizontal = {} 
        self.bb_vertical = {}

    def _compute_mixed_error(self, trend_1, trend_2, target):
        slope = trend_1 - trend_2
        prediction = trend_1 + slope
        
        # 1. Gradient Error (Prediction)
        error_pred = np.sum(np.abs(prediction - target))
        
        # 2. Color Error (Direct Match)
        error_ssd = np.sum(np.abs(trend_1 - target))
        
        return (PREDICTION_WEIGHT * error_pred) + ((1 - PREDICTION_WEIGHT) * error_ssd)

    def compute_compatibility_matrix(self):
        # Optimization: Only print once
        if SHOW_PROGRESS: print(f"   [Analysis] Pre-computing edge costs...")
        
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
        # Helper for ratio test
        def get_best(costs):
            indices = np.argsort(costs)
            best, second = indices[0], indices[1]
            ratio = costs[best] / (costs[second] + 1e-5)
            return best, ratio

        # Horizontal
        best_right = {}
        best_left = {}
        for i in range(self.num_pieces):
            j, r = get_best(self.cost_matrix[i, :, 0])
            if r < RATIO_THRESHOLD: best_right[i] = j
        for j in range(self.num_pieces):
            i, r = get_best(self.cost_matrix[:, j, 0])
            if r < RATIO_THRESHOLD: best_left[j] = i
            
        for i, j in best_right.items():
            if best_left.get(j) == i: self.bb_horizontal[i] = j

        # Vertical
        best_bottom = {}
        best_top = {}
        for i in range(self.num_pieces):
            j, r = get_best(self.cost_matrix[i, :, 1])
            if r < RATIO_THRESHOLD: best_bottom[i] = j
        for j in range(self.num_pieces):
            i, r = get_best(self.cost_matrix[:, j, 1])
            if r < RATIO_THRESHOLD: best_top[j] = i
            
        for i, j in best_bottom.items():
            if best_top.get(j) == i: self.bb_vertical[i] = j
            
        if SHOW_PROGRESS:
            print(f"   [Analysis] Robust Buddies: {len(self.bb_horizontal)} H / {len(self.bb_vertical)} V")

    def solve(self):
        self.compute_compatibility_matrix()
        self.find_robust_best_buddies()
        
        best_grid = None
        min_error = float('inf')

        print(f"   [Assembly] Testing {self.num_pieces} seeds with Strategy...")
        
        for start_node in range(self.num_pieces):
            if SHOW_PROGRESS: sys.stdout.write("."); sys.stdout.flush()
            
            # Use the NEW Frontier Assembly
            grid, score = self._assemble(start_node)
            
            if score < min_error:
                min_error = score
                best_grid = grid

        print(f"\n   [Result] Best Score: {min_error:.0f}")
        return self._render_result(best_grid)

    def _assemble(self, start_node):
        """
        Replaces the simple loop with a Priority Queue logic.
        """
        grid = np.full((self.grid_n, self.grid_n), -1, dtype=int)
        used = {start_node}
        
        # Place Seed at Top-Left (Relative placement is hard, so we assume seed is 0,0 for now)
        # Note: A true unbounded solver would place seed in center, but that requires crop logic.
        # For this dataset, iterating seeds at (0,0) is usually sufficient.
        grid[0, 0] = start_node
        
        # Frontier: List of empty slots (row, col) adjacent to placed pieces
        frontier = set()
        frontier.add((0, 1))
        frontier.add((1, 0))
        
        total_score = 0
        
        # We need to fill (Total - 1) pieces
        for _ in range(self.num_pieces - 1):
            if not frontier: break
            
            # 1. FIND MOST CONSTRAINED SLOT
            # We look for the slot in the frontier with the MAX number of filled neighbors
            best_slot = None
            max_neighbors = -1
            
            sorted_frontier = []
            
            for (r, c) in frontier:
                neighbors = 0
                if r > 0 and grid[r-1, c] != -1: neighbors += 1 # Top
                if r < self.grid_n - 1 and grid[r+1, c] != -1: neighbors += 1 # Bottom
                if c > 0 and grid[r, c-1] != -1: neighbors += 1 # Left
                if c < self.grid_n - 1 and grid[r, c+1] != -1: neighbors += 1 # Right
                
                sorted_frontier.append( (neighbors, r, c) )
            
            # Sort by neighbors descending (3 neighbors -> 2 -> 1)
            sorted_frontier.sort(key=lambda x: x[0], reverse=True)
            
            # Pick the winner
            _, target_r, target_c = sorted_frontier[0]
            
            # 2. FIND BEST PIECE FOR THIS SLOT
            best_cand = -1
            best_score = float('inf')
            
            # Identify neighbors for checking
            top_idx = grid[target_r-1, target_c] if target_r > 0 else -1
            bot_idx = grid[target_r+1, target_c] if target_r < self.grid_n - 1 else -1
            left_idx = grid[target_r, target_c-1] if target_c > 0 else -1
            right_idx = grid[target_r, target_c+1] if target_c < self.grid_n - 1 else -1
            
            for cand in range(self.num_pieces):
                if cand in used: continue
                
                score = 0
                count = 0
                is_buddy = False
                
                # Check Top Match (Candidate is Below Top)
                if top_idx != -1:
                    c = self.cost_matrix[top_idx, cand, 1]
                    if self.bb_vertical.get(top_idx) == cand:
                        c -= BEST_BUDDY_BONUS
                        is_buddy = True
                    score += c
                    count += 1
                    
                # Check Bottom Match (Candidate is Above Bottom)
                if bot_idx != -1:
                    # Note: cost_matrix is [Top][Bot], so we check [cand][bot_idx]
                    c = self.cost_matrix[cand, bot_idx, 1]
                    if self.bb_vertical.get(cand) == bot_idx:
                        c -= BEST_BUDDY_BONUS
                        is_buddy = True
                    score += c
                    count += 1
                
                # Check Left Match (Candidate is Right of Left)
                if left_idx != -1:
                    c = self.cost_matrix[left_idx, cand, 0]
                    if self.bb_horizontal.get(left_idx) == cand:
                        c -= BEST_BUDDY_BONUS
                        is_buddy = True
                    score += c
                    count += 1
                
                # Check Right Match (Candidate is Left of Right)
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
            
            # 3. PLACE & UPDATE FRONTIER
            if best_cand != -1:
                grid[target_r, target_c] = best_cand
                used.add(best_cand)
                total_score += best_score
                
                # Remove filled slot from frontier
                frontier.remove((target_r, target_c))
                
                # Add new empty neighbors to frontier
                nbs = [(target_r-1, target_c), (target_r+1, target_c), 
                       (target_r, target_c-1), (target_r, target_c+1)]
                for (nr, nc) in nbs:
                    if 0 <= nr < self.grid_n and 0 <= nc < self.grid_n:
                        if grid[nr, nc] == -1:
                            frontier.add((nr, nc))
            else:
                # If we fail to place (shouldn't happen), skip this slot
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
        
        
        save_folder = os.path.join(OUTPUT_ROOT, p_folder)
        if not os.path.exists(save_folder): os.makedirs(save_folder)
            
        cv2.imwrite(os.path.join(save_folder, "solved.jpg"), result)
        print(f"   > Saved to: {save_folder}")

if __name__ == "__main__":
    run_solver()