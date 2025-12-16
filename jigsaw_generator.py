from PIL import Image
import random
import os

our_dataset_root = 'our_dataset/correct'  
output_dataset_root = 'our_dataset'

def create_scrambled_puzzle_image(image_path, output_path, rows=4, cols=4):
    """
    Reads an image, divides it into a grid, shuffles the tiles,
    and pastes them ALL back to create a full scrambled image.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    try:
        img = Image.open(image_path)
    except IOError:
        print(f"Error: Could not open image file.")
        return

    # 1. Resize image to fit the grid perfectly.
    width, height = img.size
    tile_w = width // cols
    tile_h = height // rows
    new_width = tile_w * cols
    new_height = tile_h * rows
    
    if new_width != width or new_height != height:
        img = img.resize((new_width, new_height))
    
    # 2. Slice the image into tiles.
    tiles = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            upper = r * tile_h
            right = left + tile_w
            lower = upper + tile_h
            tile = img.crop((left, upper, right, lower))
            tiles.append(tile)

    # REMOVED: tiles.pop(blank_tile_index) 
    # (We keep all tiles now)

    # 3. Shuffle all tiles.
    random.shuffle(tiles)

    # 4. Create a new image and paste the shuffled tiles.
    new_img = Image.new('RGB', (new_width, new_height)) 
    
    tile_idx = 0
    for r in range(rows):
        for c in range(cols):
            # REMOVED: The check for blank_tile_index that skipped pasting
            
            paste_x = c * tile_w
            paste_y = r * tile_h
            
            new_img.paste(tiles[tile_idx], (paste_x, paste_y))
            tile_idx += 1

    # 5. Save the final generated image.
    new_img.save(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":

    # Ensure output directory exists
    if not os.path.exists(output_dataset_root):
        os.makedirs(output_dataset_root)

    images = [f for f in os.listdir(our_dataset_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort() # Sort to ensure consistent ordering

    if not images:
        print(f"No images found in {our_dataset_root}.")
    
    sizes = {
        "puzzle_2x2": 2,
        "puzzle_3x3": 3, 
        "puzzle_4x4": 4,
        "puzzle_6x6": 6, 
        "puzzle_8x8": 8, 
        "puzzle_10x10": 10
    }

for size_key, grid_size in sizes.items():
        
        print(f"--- Generating dataset for {size_key} ({grid_size}x{grid_size}) ---")
        
        # Enumerate gives us the index 'i' for naming the file
        for i, img_file in enumerate(images):
            
            input_path = os.path.join(our_dataset_root, img_file)
            
            # Keep original extension
            ext = os.path.splitext(img_file)[1]
            
            # Filename is simply the index (e.g., 0.jpg, 1.jpg)
            filename = f"{i}{ext}"
            
            # Output path: our_dataset/puzzle_2x2/0.jpg
            output_folder = os.path.join(output_dataset_root, size_key)
            output_path = os.path.join(output_folder, filename)
            
            # Ensure specific size folder exists
            os.makedirs(output_folder, exist_ok=True)
            
            create_scrambled_puzzle_image(input_path, output_path, rows=grid_size, cols=grid_size)
            
print("All datasets generated successfully.")