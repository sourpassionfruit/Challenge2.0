import csv
# get training data
import math
from tqdm import tqdm

# only select mid-career datasets
file = open("uci_adult.csv", encoding='utf-8-sig')
reader = csv.reader(file)
raw = []
for row in reader:
    if 40 <= int(row[0]) < 60:
        raw.append(row)

# construct 10,000 row training data and reserve 1593 rows for testing
# race (1 for white, 0 for not white), sex (1 for male, 0 for female), native country (1 for usa, 0 for others)
# income (1 for > 50k, 0 otherwise)
training = []
trainingWeights = []
for i in tqdm(range(10000)):
    race = 0
    sex = 0
    native = 0
    income = 0
    if raw[i][8] == " White":
        race = 1
    if raw[i][9] == " Male":
        sex = 1
    if raw[i][13] == " United-States":
        native = 1
    if raw[i][-1] == " >50K":
        income = 1
    training.append([race, sex, native, income])
    trainingWeights.append(int(raw[i][2]))

testing = []
testingWeights = []
for i in tqdm(range(10000, 11593)):
    race = 0
    sex = 0
    native = 0
    income = 0
    if raw[i][8] == " White":
        race = 1
    if raw[i][9] == " Male":
        sex = 1
    if raw[i][13] == " United-States":
        native = 1
    if raw[i][-1] == " >50K":
        income = 1
    testing.append([race, sex, native, income])
    testingWeights.append(int(raw[i][2]))

# writing to csv files
fields = ["Race", "Sex", "Native Country", "Income"]
with open("challenge-train.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(training)
with open("challenge-test.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(testing)

# below is naive bayes model that should work?
file = open("challenge-train.csv")
reader = csv.reader(file)
header = next(reader)
training = []
for row in reader:
    training.append(row)

# get counts and store in dictionary
xCountsY0 = {}
xCountsY1 = {}
y1Counts = 0
y0Counts = 0
for i in range(len(training[0]) - 1):
    key = "X" + str(i + 1)
    xCountsY0[key] = 0
    xCountsY1[key] = 0
j = 0  # account for density
for row in training:
    # if y = 1
    if row[-1] == "1":
        y1Counts += trainingWeights[j]
        for i in range(len(row) - 1):
            key = "X" + str(i + 1)
            xCountsY1[key] += int(row[i]) * trainingWeights[j]
    # if y = 0
    else:
        y0Counts += trainingWeights[j]
        for i in range(len(row) - 1):
            key = "X" + str(i + 1)
            xCountsY0[key] += int(row[i]) * trainingWeights[j]
    j += 1
# y0Counts = len(training) - y1Counts

# do complete laplace MAP estimates
MAPY0 = {}
MAPY1 = {}
for i in range(len(xCountsY1)):
    countKey = "X" + str(i + 1)
    # for Y = 1
    estMAPY1 = (xCountsY1[countKey] + 1) / float(y1Counts + 2)
    mapKey = countKey + "MAP"
    MAPY1[mapKey] = estMAPY1
    # for Y = 0
    estMAPY0 = (xCountsY0[countKey] + 1) / float(y0Counts + 2)
    MAPY0[mapKey] = estMAPY0

# print(MAPY0)
# print(MAPY1)

# predict testing data with model
# get testing data
file = open("challenge-test.csv")
reader = csv.reader(file)
header = next(reader)
testing = []
for row in reader:
    testing.append(row)
# predict testing data
results = []
for row in testing:
    pY1 = math.log(y1Counts / float(y0Counts + y1Counts))
    pY0 = math.log(y0Counts / float(y0Counts + y1Counts))
    l1 = pY1
    l0 = pY0
    # assume Y = 1
    for i in range(len(row) - 1):
        mapKey = "X" + str(i + 1) + "MAP"
        if row[i] == "1":
            l1 += math.log(MAPY1[mapKey])
        else:
            l1 += math.log(1 - MAPY1[mapKey])
    # assume Y = 0
    for i in range(len(row) - 1):
        mapKey = "X" + str(i + 1) + "MAP"
        if row[i] == "1":
            l0 += math.log(MAPY0[mapKey])
        else:
            l0 += math.log(1 - MAPY0[mapKey])
    # choose more likely Y, add to result
    if l1 > l0:
        results.append("1")
    else:
        results.append("0")

# check accuracy
count = 0
for i in range(len(results)):
    if results[i] == testing[i][-1]:
        count += 1
print("Accuracy is:", float(count / len(testing)))
