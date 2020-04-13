words = set()
with open('data2_cleaned.txt', 'r') as f:
    i = 0
    for line in f:
        if i == 1:
            i = 0
            tokens = line[:-1].split(' ')
            for j in tokens:
                words.add(j)
        else:
            i = 1

with open('vocab.txt', 'a') as f:
    for line in words:
        f.write(line)
        f.write('\n')
