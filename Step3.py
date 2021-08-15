# Step 3
# file feedneuralnetwork.csv will contain never draw numbers for this lotto

import datetime
import os

begin_time = datetime.datetime.now()

def shortgermanlotto():
    list_of_lists = []
    with open('germanlotto.csv', "r") as f:
        germanlotto_lines = f.readlines()
    for line in germanlotto_lines:
        sorted_line = str(sorted(map(int, line.split(','))))
        print(sorted_line)
        with open('germanlottosorted.txt', 'a') as f:
            f.write(sorted_line)
            f.write("\n")
    

    with open(r'germanlottosorted.txt', 'r') as infile, open(r'germanlottosorted.csv', 'w') as outfile:
        data = infile.read()
        data = data.replace("[", "")
        data = data.replace("]", "")
        data = data.replace(" ", "")
        outfile.write(data)
    os.remove("germanlottosorted.txt")




def compare():
    with open('germanlottosorted.csv') as f:
        germanlotto_lines = f.readlines()

    with open('combinations.csv') as f:
        combinations_lines = f.readlines()

    never_lines = [x for x in combinations_lines if x not in germanlotto_lines]

    with open('feedneuralnetwork.csv', 'w') as f:
        for s in never_lines:
            f.write(s)

#shortgermanlotto()
compare()

print("Code Executed in: ", datetime.datetime.now() - begin_time)

