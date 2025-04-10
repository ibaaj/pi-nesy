
from tools_make_examples import read_csv_file




def make_examples_sudoku(filename, puzzles):
    l = read_csv_file(filename)
    mnist_data = {}

    puzzles_map = {}

    for i in range(0, len(l)):
        mnist_data[l[i]["original_id"]] = l[i]


    for i in puzzles.keys():
        puzzle = puzzles[i]
        puzzles_map[i] = {}
        for j in range(0, len(puzzle["grid"])):
            puzzles_map[i][j] = {}
            for k in range(0, len(puzzle["grid"][j])):
                el = puzzle["grid"][j][k]
                puzzles_map[i][j][k] = mnist_data[el["list_index"]] 
                

    return puzzles_map

