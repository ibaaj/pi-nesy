import os
import glob
import numpy as np

def load_txt_files_into_dict(directory):
    all_files_dict = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    all_files_dict[os.path.splitext(file)[0]] = f.read()
    return all_files_dict

def find_split_directories(base_dir):
    split_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for dir in dirs:
            if dir.startswith("split::"):
                split_dirs.append(os.path.join(root, dir))
    return split_dirs


def parse_row_pixels(row, grid_size, image_size=28):
    pixels = list(map(float, row.strip().split()))
    images_per_row = grid_size * grid_size
    image_pixels = image_size * image_size
    grid = {}

    for i in range(grid_size):
        grid[i] = {}
        for j in range(grid_size):
            index = i * grid_size + j
            start = index * image_pixels
            end = start + image_pixels
            image = np.array(pixels[start:end]).reshape(image_size, image_size)
            grid[i][j] = image

    return grid

def parse_row_labels(row, size):
    labels = row.split()
    label_grid = []
    for i in range(size):
        row = []
        for j in range(size):
            
            number = labels[i * size + j].split('_')[1]
            row.append(int(number))
        label_grid.append(row)
    return label_grid

def parse_puzzles(files_dict, size, split_type): 
    # split_type :train, valid, test 
    index_puzzle = 0
    puzzles = {}
    for puzzle_row in files_dict[f"{split_type}_puzzle_labels"].split('\n'):
        if puzzle_row.strip():
            truth = 1 if int(puzzle_row[0]) == 1 else 0
            puzzles[index_puzzle] = {'puzzle_id': index_puzzle, 'truth': truth, 'grid': {}, 'size': size}
            index_puzzle+=1
    index_puzzle = 0
    index_image = 0
    for row_label, row_images in zip(files_dict[f"{split_type}_cell_labels"].split('\n'), files_dict[f"{split_type}_puzzle_pixels"].split('\n')):
        if row_label.strip() and row_images.strip():
            grid = parse_row_pixels(row_images, size)
            label_grid = parse_row_labels(row_label, size)
            for i in range(size):
                puzzles[index_puzzle]["grid"][i] = {}
                for j in range(size):
                    puzzles[index_puzzle]["grid"][i][j] = {'image_data': grid[i][j], 'truth_label': label_grid[i][j], 'list_index': index_image}
                    index_image+=1
            index_puzzle+=1
            
    return puzzles