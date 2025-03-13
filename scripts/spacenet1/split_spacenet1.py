import os
import glob
import random
import shutil
import re
import argparse

def extract_id(filename):
    """Extract the image ID from a filename."""
    match = re.search(r'img(\d+)', filename)
    if match:
        return match.group(1)
    return None

def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def main(train_dir, val_percentage, test_percentage):
    # Define source and destination directories
    val_dir = os.path.join(os.path.dirname(train_dir), 'val')
    test_dir = os.path.join(os.path.dirname(train_dir), 'test')
    
    # Create val and test directories and subdirectories
    for target_dir in [val_dir, test_dir]:
        create_directory(target_dir)
        for subdir in ['geojson', '3band', '8band']:
            create_directory(os.path.join(target_dir, subdir))
    
    # Collect all unique IDs
    all_ids = set()
    
    # Check geojson files
    geojson_files = glob.glob(os.path.join(train_dir, 'geojson', 'Geo_AOI_1_RIO_img*.geojson'))
    for file in geojson_files:
        img_id = extract_id(os.path.basename(file))
        if img_id:
            all_ids.add(img_id)
    
    # Check 3band files
    band3_files = glob.glob(os.path.join(train_dir, '3band', '3band_AOI_1_RIO_img*.tif'))
    for file in band3_files:
        img_id = extract_id(os.path.basename(file))
        if img_id:
            all_ids.add(img_id)
    
    # Check 8band files
    band8_files = glob.glob(os.path.join(train_dir, '8band', '8band_AOI_1_RIO_img*.tif'))
    for file in band8_files:
        img_id = extract_id(os.path.basename(file))
        if img_id:
            all_ids.add(img_id)
    
    # Convert to list for random selection
    all_ids = list(all_ids)
    print(f"Found {len(all_ids)} unique image IDs")
    
    # Calculate number of IDs to move
    num_val = int(len(all_ids) * val_percentage / 100)
    num_test = int(len(all_ids) * test_percentage / 100)
    print(f"Moving {num_val} IDs ({val_percentage}%) to validation set")
    print(f"Moving {num_test} IDs ({test_percentage}%) to test set")
    
    # Randomly select IDs to move
    random.shuffle(all_ids)
    val_ids = all_ids[:num_val]
    test_ids = all_ids[num_val:num_val+num_test]
    
    # Function to move files for a set of IDs to a target directory
    def move_files_for_ids(ids_to_move, target_dir):
        moved_count_geojson = 0
        moved_count_band3 = 0
        moved_count_band8 = 0
        for img_id in ids_to_move:
            # Move geojson file
            geojson_src = os.path.join(train_dir, 'geojson', f'Geo_AOI_1_RIO_img{img_id}.geojson')
            geojson_dst = os.path.join(target_dir, 'geojson', f'Geo_AOI_1_RIO_img{img_id}.geojson')
            if os.path.exists(geojson_src):
                shutil.move(geojson_src, geojson_dst)
                moved_count_geojson += 1
            
            # Move 3band file
            band3_src = os.path.join(train_dir, '3band', f'3band_AOI_1_RIO_img{img_id}.tif')
            band3_dst = os.path.join(target_dir, '3band', f'3band_AOI_1_RIO_img{img_id}.tif')
            if os.path.exists(band3_src):
                shutil.move(band3_src, band3_dst)
                moved_count_band3 += 1
            
            # Move 8band file
            band8_src = os.path.join(train_dir, '8band', f'8band_AOI_1_RIO_img{img_id}.tif')
            band8_dst = os.path.join(target_dir, '8band', f'8band_AOI_1_RIO_img{img_id}.tif')
            if os.path.exists(band8_src):
                shutil.move(band8_src, band8_dst)
                moved_count_band8 += 1
        
        return moved_count_geojson, moved_count_band3, moved_count_band8
    
    # Move files to validation and test directories
    val_moved_geojson, val_moved_band3, val_moved_band8 = move_files_for_ids(val_ids, val_dir)
    test_moved_geojson, test_moved_band3, test_moved_band8 = move_files_for_ids(test_ids, test_dir)
    
    print(f"Moved {val_moved_geojson} geojson files to validation directory")
    print(f"Moved {test_moved_geojson} geojson files to test directory")
    print(f"Moved {val_moved_band3} 3band files to validation directory")
    print(f"Moved {test_moved_band3} 3band files to test directory")
    print(f"Moved {val_moved_band8} 8band files to validation directory")
    print(f"Moved {test_moved_band8} 8band files to test directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training, validation, and test sets')
    parser.add_argument('train_dir', type=str, help='Path to the training directory')
    parser.add_argument('--val_percentage', type=float, default=10.0, 
                        help='Percentage of files to move to validation (default: 10.0)')
    parser.add_argument('--test_percentage', type=float, default=20.0, 
                        help='Percentage of files to move to test (default: 10.0)')
    
    args = parser.parse_args()
    
    main(args.train_dir, args.val_percentage, args.test_percentage)
