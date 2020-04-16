import pandas as pd
import numpy as np
import string

data_arr = []
with open('data_train.txt', 'r') as f:

    i = 0
    entry = dict()
    columns = ['flair', 'title']

    # read file to create list of dicts
    for line in f:
        entry[columns[i]] = " ".join(line[:-1].split())
        i+=1
        # end of one entry
        if i == 2:
            data_arr.append(entry)
            entry = dict()
            i = 0
#print(data_arr)

df = pd.DataFrame(data_arr)
valid_flairs = ['Politics', 'Non-Political', 'AskIndia', 'Policy/Economy','Business/Finance','Science/Technology', 'Scheduled', 'Sports', 'Food','Photography','CAA-NRC-NPR', 'Coronavirus']

map_int = dict()
for i, j in enumerate(valid_flairs):
    map_int[j] = i

# Count no of data points under all labels
label_count = []
for i in valid_flairs:
    label_count.append((df['flair'] == i).sum())

# Pretty print the data
for i, j in enumerate(valid_flairs):
    print("{}: ".format(j), end='')
    print(label_count[i], end='')
    print(" of {}".format(df.shape[0]))

df = df.sample(frac=1).reset_index(drop=True)

# Write out data
# get no of rows for each label to be present in validation set
label_count_small = [x//350 for x in label_count]
label_count_curr = [0 for x in label_count]

# iterate over whole dataset
for index,row in df.iterrows():
    cur_index = map_int[row['flair']]
    # if current count exceeds needed for validation, write to train
    if label_count_curr[cur_index] > label_count_small[cur_index]:
        #with open('data_train.txt', 'a') as f:
        print('ow2')
        #    f.write(row['flair'])
        #    f.write('\n')
        #    f.write(process_text(row['title']))
        #    f.write('\n')
    # else write to validation
    else:
        with open('data_small_tiny.txt', 'a') as f:
            print('ow')
            f.write(row['flair'])
            f.write('\n')
            f.write(row['title'])
            f.write('\n')
            label_count_curr[cur_index] += 1
