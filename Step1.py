# Step N 1

# Creates file with all unique posible combinations
from itertools import combinations
import datetime
import os

begin_time = datetime.datetime.now()

numbersinlotto = 49
numbersinresults = 6

f = open('combinations.txt', 'w')
for comb in combinations(range(1,numbersinlotto+1), numbersinresults):
    f.write(str(comb))
    f.write('\n')

f.close()


# Clean file from junk and create final CSV file
with open(r'combinations.txt', 'r') as infile, open(r'combinations.csv', 'w') as outfile:
    data = infile.read()
    data = data.replace("(", "")
    data = data.replace(")", "")
    outfile.write(data)
os.remove("combinations.txt")




print("Code Executed in: ",datetime.datetime.now() - begin_time)
