# Computer Vision Jigsaw Puzzle Solver

## Project Overview
This project implements an automated Jigsaw Puzzle Solver using classical Computer Vision techniques and mathematical heuristics. It is designed to solve digital scrambled grid puzzles (2x2, 4x4, and 8x8) **without** the use of Machine Learning, or AI training data.

The system relies on deterministic algorithms such as **Linear Prediction** (for gradient matching), **Best Buddy** analysis (graph theory), and **Beam Search** (optimization) to reconstruct images based purely on edge continuity and pixel statistics.

## Project Structure
The solution is divided into two strict milestones:

- **Phase 1: Preprocessing & Extraction** (Preparing the data)
- **Phase 2: Solver & Assembly** (Reconstructing the image)

## Prerequisites
- Python 3.8+
- OpenCV (`cv2`)
- NumPy


## Dataset Organization
The project expects the raw dataset to be organized in the root directory as follows:

```text
/Project_Root
	 |-- phase1_preprocessor.py      # The Phase 1 Script
	 |-- phase2_solver.py            # The Phase 2 Script 
	 |-- Jigsaw Puzzle Dataset/      # Dataset
			 |-- Gravity Falls/
					 |-- puzzle_2x2/
					 |-- puzzle_4x4/
					 |-- puzzle_8x8/
							 |-- 0.jpg
							 |-- ...
```

## Phase 1: Preprocessing & Dataset Preparation

### Objective
Transform raw grid images into a normalized dataset of individual, perfectly square digital assets.

### 1. Key Techniques Used
- **Grid Slicing**: The script mathematically calculates the exact dimensions of each cell based on the input image resolution and grid size (2, 4, or 8).
- **Normalization**: All extracted pieces are resized to a standard 100x100 pixel resolution. This ensures the solver compares consistent feature vectors regardless of the original image size.
- **Image Enhancement Pipeline**:
	- **Denoising**: Applies a Gaussian Blur (3,3) to remove JPEG compression artifacts.
	- **Sharpening**: Applies a Laplacian-based kernel ([-1, 5, -1]) to boost edge contrast. This is critical for making boundary pixels distinct for the solver.

### 2. How to Run Phase 1
Run the preprocessor script. It will automatically detect 2x2, 4x4, and 8x8 folders.


### 3. Deliverables (Output)
The script creates a folder named `processed_dataset/`.

- **Full Reference**: A `full_enhanced.jpg` is saved for visual verification.
- **Assets**: Individual pieces are saved as `piece_000.png`, `piece_001.png`, etc.

## Phase 2: The Solver & Assembly

### Objective
Reassemble the shuffled normalized pieces into the correct image using “Prediction-Based” greedy and search algorithms.

### 1. Key Algorithms & Logic
The solver uses a sophisticated **Edge Compatibility** metric rather than simple color matching.

- **Color Space Conversion**: Pieces are converted from BGR to LAB Color Space to match human color perception.
- **Mathematical Metric: Linear Prediction (Gradient Extrapolation)**:
	- Instead of checking if $Pixel_A \approx Pixel_B$, the solver calculates the gradient (slope) of Piece A’s edge.
	- It predicts what the next pixel should be and compares that prediction to Piece B.
	- **Benefit**: This allows the solver to correctly assemble gradients (e.g., a sky fading from dark to light blue) where simple color matching fails.
- **Best Buddies Algorithm**:
	- Identifies pairs of pieces that are “Mutually Best Matches” (Piece A wants B, and B wants A).
	- These are locked together first to form stable “islands.”
- **Advanced Search Strategy**:
	- **Beam Search (for 8x8)**: Maintains multiple parallel timelines (top K solutions) to recover from early errors.
	- **Consistency Checking**: Before merging large clusters, the solver verifies that all touching edges are compatible, not just the primary connection.

### 2. How to Run Phase 2
Run the solver script. It will iterate through the processed pieces and generate solutions.


### 3. Deliverables (Output)
The script creates a folder named `solved_puzzles/` (or `solved_puzzles_beam_8x8` depending on configuration).

- **Solution Image**: A single JPEG file (e.g., `solution.jpg`) showing the fully reconstructed puzzle.
- **Console Logs**: Detailed output showing the algorithm’s confidence scores, “Best Buddy” pairs found, and the final error metric.
