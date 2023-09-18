import sys

def read_file_and_sort_tuples(file_path):
    with open(file_path, 'r') as file:
        rows = file.readlines()

    tuples = [tuple(map(int, row.strip().split())) for row in rows]

    sorted_tuples = sorted(tuples)

    return sorted_tuples

file1 = sys.argv[1]
file2 = sys.argv[2]

t1 = read_file_and_sort_tuples(file1)
t2 = read_file_and_sort_tuples(file2)

allSame = True
for i, j in zip(t1, t2):
  if i != j:
    allSame = False

print(f'all Same: {allSame}')