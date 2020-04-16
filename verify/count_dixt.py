#files = []
valid_flairs = ['Politics', 'Non-Political', 'AskIndia', 'Policy/Economy', 'Business/Finance', 'Science/Technology', 'Scheduled', 'Sports', 'Food', 'Photography', 'CAA-NRC-NPR', 'Coronavirus']

counts = [0 for i in range(len(valid_flairs))]
map_int = dict()
    
for i, j in enumerate(valid_flairs):
    map_int[j] = i


i = 0
with open('data_correct_med.txt', 'r') as f:
    for line in f:
            if i == 0:
                #print('output: ', line)
                #load_output.append(map_int[line[:-1]])
                counts[map_int[line[:-1]]] += 1
                i = 1


            elif i == 1:
                # truncate sequence to 100 length
                #print('input: ', line)
                #cur_input = []
                
                #len_in = len(line[:-1].split()[:25])
                #for i in range(25-len_in):
                #    cur_input.append(' ')
                #for i in line[:-1].split()[:25]:
                #    cur_input.append(i)
                #load_input.append(cur_input)
                i = 0
print(counts)
print(sum(counts))
