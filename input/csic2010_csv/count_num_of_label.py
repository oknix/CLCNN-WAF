import csv

count = 0
count2 = 0
with open("anomalousTrafficTest.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row[1]) < 100:
            print(row[1])
            count2 += 1
        count += 1
print(count)
print(count2)

count = 0
count2 = 0

with open("normalTrafficTraining.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row[1]) < 100:
            print(row[1])
            count2 += 1
        count += 1
print(count)
print(count2)
