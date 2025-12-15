import cv2
import numpy as np
import os

# --- 1. CONFIGURATION ---
# The folder where your raw images are stored
DATASET_ROOT = "Jigsaw Puzzle Dataset" 

# The folder where processed assets will be saved
OUTPUT_ROOT = "processed_dataset"

# Target resolution for the individual pieces (Phase 2 requirement)
TARGET_SIZE = 100 

# Map folder names to their grid dimensions
FOLDERS_TO_PROCESS = {
    "puzzle_2x2": 2,
    "puzzle_4x4": 4,
    "puzzle_8x8": 8
}

def enhance_image(image):
    """
    Applies image enhancement pipeline:
    1. Gaussian Blur to remove compression noise.
    2. Sharpening Kernel to make edges distinct.
    """
    # Step A: Mild Denoising
    # (3,3) kernel size is gentle enough not to blur important details
    denoised = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Step B: Sharpening
    # This matrix boosts the center pixel relative to neighbors
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return sharpened

def process_puzzle(img_path, grid_size, category_name):
    """
    Loads one puzzle image, enhances it, saves the full version,
    then slices it into normalized pieces.
    """
    # 1. Load Original Image
    img = cv2.imread(img_path)
    if img is None:
        print(f"   [Error] Could not read file: {img_path}")
        return

    # Get file ID (e.g., '0' from '0.jpg')
    puzzle_id = os.path.splitext(os.path.basename(img_path))[0]
    
    # Define output directory structure:
    # processed_dataset / Gravity Falls / puzzle_4x4 / puzzle_0 /
    folder_type = f"puzzle_{grid_size}x{grid_size}"
    save_dir = os.path.join(OUTPUT_ROOT, category_name, folder_type, f"puzzle_{puzzle_id}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- 2. ENHANCE & SAVE FULL IMAGE (User Request) ---
    # We create a full-size enhanced copy so you can see what the 'perfect' puzzle looks like.
    full_enhanced = enhance_image(img)
    cv2.imwrite(os.path.join(save_dir, "full_enhanced.jpg"), full_enhanced)

    # --- 3. SLICE & NORMALIZE PIECES ---
    height, width, _ = img.shape
    
    # Calculate dimension of each cell
    step_y = height // grid_size
    step_x = width // grid_size

    piece_idx = 0
    for y in range(grid_size):
        for x in range(grid_size):
            start_y = y * step_y
            start_x = x * step_x
            
            # Extract piece from the ORIGINAL image
            # (Resizing the raw pixels then sharpening is better than sharpening then resizing)
            raw_piece = img[start_y : start_y + step_y, start_x : start_x + step_x]
            
            # A. Normalize Size (100x100)
            normalized = cv2.resize(raw_piece, (TARGET_SIZE, TARGET_SIZE))
            
            # B. Enhance the small piece
            final_piece = enhance_image(normalized)
            
            # Save the individual asset
            filename = f"piece_{piece_idx:03d}.png"
            cv2.imwrite(os.path.join(save_dir, filename), final_piece)
            
            piece_idx += 1
            
    print(f"   > Processed: {puzzle_id}.jpg -> Saved {piece_idx} pieces + full_enhanced.jpg")

def run_pipeline():
    print(f"--- Phase 1: Preprocessing Pipeline ---")
    
    # Check if dataset root exists
    if not os.path.exists(DATASET_ROOT):
        print(f"[CRITICAL ERROR] Folder '{DATASET_ROOT}' not found.")
        print("Please make sure the script is running next to the 'Jigsaw Puzzle Dataset' folder.")
        return

    found_work = False

    # Walk through the entire dataset directory
    for root, dirs, files in os.walk(DATASET_ROOT):
        current_folder = os.path.basename(root)
        
        # Check if we are in a target folder (puzzle_2x2, 4x4, etc.)
        if current_folder in FOLDERS_TO_PROCESS:
            grid_size = FOLDERS_TO_PROCESS[current_folder]
            
            # Identify Category (e.g., 'Gravity Falls' is the parent of 'puzzle_4x4')
            category_name = os.path.basename(os.path.dirname(root))
            
            print(f"\n[+] Found Group: {category_name} / {current_folder}")
            
            # Filter for images only
            valid_images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not valid_images:
                print("    [!] Folder is empty.")
                continue
                
            found_work = True
            
            # Process every image in this folder
            for img_file in valid_images:
                full_path = os.path.join(root, img_file)
                process_puzzle(full_path, grid_size, category_name)

    if not found_work:
        print("\n[Warning] No matching folders (puzzle_2x2, puzzle_4x4, puzzle_8x8) found.")
    else:
        print("\n--- Pipeline Complete. Dataset is ready for Phase 2. ---")

if __name__ == "__main__":
    run_pipeline()