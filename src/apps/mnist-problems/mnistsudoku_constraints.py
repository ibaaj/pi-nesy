import math




def sudoku_build_set_constraints(size):
    constraints = []

    for i in range(0, size):
        for j in range(0, size):
            for jp in range(j + 1, size):
                constraints.append((i, j, i, jp))

    for j in range(0, size):
        for i in range(0, size):
            for ip in range(i + 1, size):
                constraints.append((i, j, ip, j))

    for S in range(0, size):
        for i in range(0, size):
            for ip in range(i + 1, size):
                if size == 4:
                    rsi = int(2 * (S / 2) + math.floor(i / 2) - S % 2)
                    csi = int((i % 2) + ((S % 2) * 2))
                    rsip = int(2 * (S / 2) + math.floor(ip / 2) - S % 2)
                    csip = int((ip % 2) + ((S % 2) * 2))
                else:  # sudoku 9x9
                    rsi = int(3 * (S / 3) + math.floor(i / 3) - S % 3)
                    csi = int((i % 3) + ((S % 3) * 3))
                    rsip = int(3 * (S / 3) + math.floor(ip / 3) - S % 3)
                    csip = int((ip % 3) + ((S % 3) * 3))
                if (rsi, csi, rsip, csip) not in constraints:
                    constraints.append((rsi, csi, rsip, csip))

    return constraints