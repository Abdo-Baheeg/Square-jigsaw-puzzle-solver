import cv2
import numpy as np
import os
import glob
import sys
import heapq
import time

# --- CONFIGURATION ---
INPUT_ROOT = "processed_dataset"
OUTPUT_ROOT = "solved_puzzles_beam_8x8"

# 8x8 Specific Tuning
GRID_SIZE = 8
BEAM_WIDTH = 200        # Keep top 200 solutions alive (Higher = Slower but Smarter)
TRIM_MARGIN = 3         # Ignore corner noise
PREDICTION_WEIGHT = 0.7 # 70% Gradient Logic

class Piece:
    def __init__(self, image_path, index):
        self.index = index
        self.bgr = cv2.imread(image_path)
        self.h, self.w, _ = self.bgr.shape
        self.lab = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2LAB).astype("float32")

    def get_edge(self, orientation):
        # 0:Top, 1:Bottom, 2:Left, 3:Right
        if orientation == 0: return self.lab[0, TRIM_MARGIN:-TRIM_MARGIN]
        if orientation == 1: return self.lab[-1, TRIM_MARGIN:-TRIM_MARGIN]
        if orientation == 2: return self.lab[TRIM_MARGIN:-TRIM_MARGIN, 0]
        if orientation == 3: return self.lab[TRIM_MARGIN:-TRIM_MARGIN, -1]
        
    def get_pixel_probe(self, orientation):
        if orientation == 0: return self.lab[1, TRIM_MARGIN:-TRIM_MARGIN]
        if orientation == 1: return self.lab[-2, TRIM_MARGIN:-TRIM_MARGIN]
        if orientation == 2: return self.lab[TRIM_MARGIN:-TRIM_MARGIN, 1]
        if orientation == 3: return self.lab[TRIM_MARGIN:-TRIM_MARGIN, -2]

class BeamSolver8x8:
    def __init__(self, piece_files):
        self.pieces = [Piece(f, i) for i, f in enumerate(piece_files)]
        self.num_pieces = len(self.pieces)
        
        # Validation
        if self.num_pieces != 64:
            print(f"[Warning] Folder has {self.num_pieces} pieces, but this solver is strictly for 8x8 (64 pieces).")
        
        self.cost_matrix = np.zeros((self.num_pieces, self.num_pieces, 2), dtype=np.float32)

    def _calc_score(self, edge1, inner1, edge2):
        slope = edge1 - inner1
        prediction = edge1 + slope
        grad_error = np.sum(np.abs(prediction - edge2))
        ssd_error = np.sum(np.abs(edge1 - edge2))
        return (PREDICTION_WEIGHT * grad_error) + ((1 - PREDICTION_WEIGHT) * ssd_error)

    def precompute_costs(self):
        print(f"   [Analysis] Pre-calculating all connection costs...")
        for i in range(self.num_pieces):
            for j in range(self.num_pieces):
                if i == j: 
                    self.cost_matrix[i, j, :] = float('inf')
                    continue
                
                pA, pB = self.pieces[i], self.pieces[j]
                
                # Horizontal (A Left of B)
                self.cost_matrix[i, j, 0] = self._calc_score(
                    pA.get_edge(3), pA.get_pixel_probe(3), pB.get_edge(2)
                )
                
                # Vertical (A Top of B)
                self.cost_matrix[i, j, 1] = self._calc_score(
                    pA.get_edge(1), pA.get_pixel_probe(1), pB.get_edge(0)
                )

    def solve(self):
        self.precompute_costs()
        print(f"   [Assembly] Starting Beam Search (Width={BEAM_WIDTH})...")

        # Beam State: (Cost, Grid_Tuple, Used_Set_Tuple)
        # We use Tuples because lists are not hashable/comparable for sets
        
        # 1. SEEDING: Try every piece at (0,0)
        beam = []
        for pid in range(self.num_pieces):
            # Grid is flattened tuple of length 64 (indices)
            # -1 represents empty
            grid = [-1] * 64
            grid[0] = pid
            
            used = {pid}
            heapq.heappush(beam, (0, tuple(grid), tuple(sorted(list(used)))))

        # We assume standard scanline order filling for simplicity in this version:
        # 0, 1, 2, 3 ... 7
        # 8, 9 ...
        # This reduces search complexity vs "Frontier" search.
        
        for step in range(1, 64): # We need to fill indices 1 to 63
            sys.stdout.write(f"\r   > Placing Piece {step+1}/64 | Beam: {len(beam)} candidates")
            sys.stdout.flush()
            
            next_beam = []
            
            # Identify target slot (r, c)
            target_idx = step
            r, c = divmod(target_idx, 8)
            
            # EXPAND
            # We take the top K candidates from previous step
            best_candidates = heapq.nsmallest(BEAM_WIDTH, beam, key=lambda x: x[0])
            
            for score, grid_tuple, used_tuple in best_candidates:
                grid = list(grid_tuple)
                used = set(used_tuple)
                
                # Identify neighbors that constrain this slot
                # Left Neighbor (index - 1)
                left_pid = grid[target_idx - 1] if c > 0 else -1
                # Top Neighbor (index - 8)
                top_pid = grid[target_idx - 8] if r > 0 else -1
                
                # Try all unused pieces
                for cand in range(self.num_pieces):
                    if cand in used: continue
                    
                    added_cost = 0
                    
                    # Cost from Left
                    if left_pid != -1:
                        added_cost += self.cost_matrix[left_pid, cand, 0]
                        
                    # Cost from Top
                    if top_pid != -1:
                        added_cost += self.cost_matrix[top_pid, cand, 1]
                        
                    # LOOKAHEAD HEURISTIC (The "Enhancement")
                    # If we place 'cand' here, does it kill the chance for the RIGHT neighbor?
                    # (Only check if not at right edge)
                    if c < 7:
                        # Find the best possible piece for the *next* slot (right side)
                        min_future_cost = float('inf')
                        for future_cand in range(self.num_pieces):
                            if future_cand not in used and future_cand != cand:
                                cost = self.cost_matrix[cand, future_cand, 0]
                                if cost < min_future_cost:
                                    min_future_cost = cost
                        
                        # If the best future move is terrible, penalize this move now
                        if min_future_cost > 5000: # Arbitrary high threshold
                             added_cost += min_future_cost # Add penalty
                    
                    new_score = score + added_cost
                    
                    # Create new state
                    new_grid = list(grid)
                    new_grid[target_idx] = cand
                    new_used = list(used)
                    new_used.append(cand)
                    
                    heapq.heappush(next_beam, (new_score, tuple(new_grid), tuple(sorted(new_used))))
            
            # Prune beam
            beam = heapq.nsmallest(BEAM_WIDTH, next_beam, key=lambda x: x[0])

        print("\n   [Result] Search complete.")
        best_solution = beam[0] # (Score, Grid, Used)
        return self._render(best_solution[1])

    def _render(self, grid_tuple):
        piece_size = self.pieces[0].h
        canvas = np.zeros((piece_size * 8, piece_size * 8, 3), dtype="uint8")
        
        for idx, pid in enumerate(grid_tuple):
            if pid == -1: continue
            r, c = divmod(idx, 8)
            y, x = r * piece_size, c * piece_size
            canvas[y:y+piece_size, x:x+piece_size] = self.pieces[pid].bgr
            
        return canvas

def run_beam_8x8():
    if not os.path.exists(INPUT_ROOT): return
    
    # Only look for puzzle_8x8 folders
    target_folders = []
    for root, dirs, files in os.walk(INPUT_ROOT):
        if "puzzle_8x8" in root and "piece_000.png" in files:
            target_folders.append(root)

    print(f"Found {len(target_folders)} 8x8 puzzles.")

    for p_folder in target_folders:
        print(f"\n--- Solving 8x8: {os.path.basename(p_folder)} ---")
        files = sorted(glob.glob(os.path.join(p_folder, "piece_*.png")))
        if not files: continue

        solver = BeamSolver8x8(files)
        result = solver.solve()
        
        rel_path = os.path.relpath(p_folder, INPUT_ROOT)
        save_folder = os.path.join(OUTPUT_ROOT, rel_path)
        if not os.path.exists(save_folder): os.makedirs(save_folder)
            
        cv2.imwrite(os.path.join(save_folder, "solution_beam_8x8.jpg"), result)
        print(f"   > Saved to: {save_folder}")

if __name__ == "__main__":
    run_beam_8x8()