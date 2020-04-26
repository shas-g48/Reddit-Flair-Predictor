# Model Analysis

## Performance:

* At training accuracy ~75 % has 'validation accuracy ~60.7' % after training for 73 epochs
* This is trained on data7 (final.txt) whoch contains all data scraped except data7 (test.txt)
* It does not do much for caa-nrc-npr as the data was sparse in my dataset

## Raw Log:

```
Test Performance:
No of data points per category: [3783, 3506, 1919, 766, 1011, 646, 81, 280, 157, 214, 4, 309]
#debug time to eval:  2.1835451126098633
#debug got total correct 7698 out of 12676:
#debug correct by category:  [[2700 2164 1121  279  622  273   54  176   71   73    0  165]]
#debug % accuracy: 0.6072893657305144

Train Performance:
time to train one epoch 73 : 43.41581726074219
average loss at epoch 73 : 0.728864552661977
#debug time to eval:  18.469125747680664
#debug got total correct 85572 out of 114147:
#debug correct by category:  [[27979 24103 12333  3753  7054  3684   548  2070   938  1103     7  2000]]
#debug % accuracy: 0.7496649057793897

#debug epoch: 73 no of batches: 892
```

## Performance by flair:
Politics: 71.3
Non-Political: 61.7
AskIndia: 58.4
Policy/Economy: 36.4
Business/Finance: 61.5
Science/Technology: 42.2
Scheduled: 66.7
Sports: 62.8
Food: 45.2
Photography: 34.1
CAA-NRC-NPR: 0 (lack of data points in test set)
Coronavirus: 53.4

## What is it good at?

* Detecting flairs with lots of data points like politics
* Detecting flairs having titles with lots of strucute like scheduled. (This confirms my eda analysis of flair)

## What is it bad at?

* Detecting flairs with less data, like caa-nrc-npr
* Distinguishing between flairs that are related, like caa-nrc-npr and politics.
* As my model only takes in 50 words, it is bad at titles having the actual content beyond 50 words.
* It is bad at posts having more signal in self text, in which the title itself is not adequate in finding the flair
* It does not see comments, so it is bad at detecting when the actual relevant info is in the comments
* I ignored both above based on my eda analysis
* It may suffer due to bad embeddings that are just trained on the scraped data

## What is the performance compared to baselines?
* Input Independent baseline (when I feed in zeros instead of data, but feed in the labels) I get an accuracy of 18.5 % on the train set. (See experiment logs/raw/experiment_log.txt 8-2)
* So atleast the model actually learns something form the data.

## What other issues could be there?
* I evaluated myself on the data points the model got wrong, and got 71 wrong out of 120. This is a very small evaluation size but this amounts to 41 % accuracy. (See experiment logs/raw/experiment_log.txt Others: 2.)
* I found issues with the dataset, many points that I predicted caa-nrc-npr were actually tagged politics, which I found very confusing
* Further many policy economy wer mixed with politics and non political
* A lot of posts contain things other than text like images, videos which the model cannot see
* I had to do a lot of guess work, some examples are below

```
Mumbai continues to show up
>> Unclear, is it coronavirus, non-political or political
Having a great time
>> Can be non political if posted on personal experience, but changes if satirical
Main Bhi Bharat Tribes of India
>> (looks like a documentary maybe)
Maunendra
>> What is this?
Is this really whats happening
>> Which event is this referring to?
Is it only the common people who have to abide by the rules boundaries and whatever
>> Coronavirus or caa-nrc-npr or politics? (should not be askindia, can be coronavirus if recent, but still go with politics)
Right
>> One word, no interpretable meaning
This is me the guy who left home
>> I had a high confidence that this was non-political, but turned out to be photography
For those who are in a dilemma on what to do after th
>> Text processing removed some useful info?
India celebrating a new festival or it s just a distraction
>>  I had a high confidence this was coronavirus (candle lighting done nationwide), but this was tagged politics
```

* I do not expect the model to get these right, this is noise in the dataset.

## Did you try regularization?
* I tried regularizing, but couldn't get the performance to go beyond ~65 % on the validation set.
* Validation accuracy goes up, reaches a peak, and goes down. I used the criteria to stop training if validation does not exceed last max for 8 epochs (later set to 15 epochs)

## Loss Curves from some attempts at regularization:


## Raw log of regularization attempts
```
===================
regular-1

Train:
#debug got total correct 50797 out of 76111: 
#debug correct by category:  [[18984 12953  7497  1763  3772  1876   292  1258   499   555     0  1348]]
#debug % accuracy: 0.6674068137325748
average loss at epoch 30 : 1.064738113719064

Test:
#debug got total correct 16036 out of 25360: 
#debug correct by category:  [[6136 4100 2387  491 1190  540   86  372  143  152    0  439]]
#debug % accuracy: 0.6323343848580442

params:
batchsize = 32
l2 regularization for gru cell (kern = 0 rec = 0)
dense 1 l2 d1w = 0.6
dense 1 drop d1dr = 0.5
dense 2 l2 d2w = 0.6
dense 2 drop d2dr = 0.5
last out l2 lsdr = 0.5
last out l2 logw = 0.4
max validation maxv = 63.6

==================
regular-2

Train:
#debug got total correct 50725 out of 76111: 
#debug correct by category:  [[18460 12577  7772  2217  3884  1840   291  1267   491   551     0  1375]]
#debug % accuracy: 0.6664608269501123
average loss at epoch 28 : 1.0750384295845892

Test:
#debug got total correct 16028 out of 25360: 
#debug correct by category:  [[5963 3977 2506  659 1188  536   85  372  140  161    0  441]]
#debug % accuracy: 0.6320189274447949

params:
l2 regularization for gru cell cell kern = 0.4 rec = 0.2
max validation maxv = 63.55
other params kept constant
===================
regular-3

Train:
#debug got total correct 52208 out of 76111: 
#debug correct by category:  [[18354 15194  7237  1686  3996  1921   312  1280   445   446     5  1332]]
#debug % accuracy: 0.6859455269277765
average loss at epoch 19 : 0.9052819416651855

Test:
#debug got total correct 16044 out of 25360: 
#debug correct by category:  [[5869 4624 2258  463 1191  538   90  364  124  114    0  409]]
#debug % accuracy: 0.6326498422712934

train with all regularization removed
params: all zero
max validation maxv: 63.888
=======================
regular-4

Train:
#debug got total correct 53528 out of 76111: 
#debug correct by category:  [[18568 14387  7770  2249  4195  2271   333  1262   568   547     4  1374]]
#debug % accuracy: 0.7032886179395882
average loss at epoch 24 : 0.8710121667492497

Test:
#debug got total correct 15923 out of 25360: 
#debug correct by category:  [[5832 4215 2348  584 1214  599   93  348  146  133    0  411]]
#debug % accuracy: 0.627878548895899

search over l2 norm + increase stopping criteria to 15

all l2 = 0.1
maxv = 64.03
=======================
regular-5

Train:
#debug got total correct 53766 out of 76111: 
#debug correct by category:  [[19425 13845  7875  1881  3989  2402   323  1322   547   791     6  1360]]
#debug % accuracy: 0.7064156298038391
average loss at epoch 28 : 0.8426924246173721

Test:
#debug got total correct 15700 out of 25360: 
#debug correct by category:  [[6005 3950 2399  479 1095  611   92  366  123  174    0  406]]
#debug % accuracy: 0.6190851735015773

parmas:
all l2 = 0.6
```
