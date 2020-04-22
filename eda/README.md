# Exploratory data analysis on data scraped from r/india

## Run
```
jupyter notebook
then run all cells
```

## File info
* eda.ipynb: eda jupyter notebook
* data5.txt: sample data


## Here is a overview of the analysis done

### Keep data with only valid flairs
1. Clean up the scraped data
1. It is assumed that the predictor only needs to deal with the currently valid flairs on r/india

### Find number of empty values in each column
1. Find number of empty values in each column, so as to decide what to include in the input
1. Found that a lot of selftext is empty (102k of 126k), this may be due to it being a media file / no alphabetical data
1. Comments have a increasing order, with top level comment 1 having the lowest empty rate and top level comment 3 having the most empty entries (52k, 70k, 82k respectively)
1. The signal to noise ratio should also be highest for title, selftext, then comments
1. Some titles were also found empty, they were removed. (must have happened due to preprocessing)

### Count number of datapoints by label
1. Find the data distribution across categories, which flairs dominate and which are less present in the dataset
1. Found data for caa-nrc-npr and scheduled meagre (< 1000)
1. Data for sports, food, photography and coronavirus is (< 3500)
1. Policy/economy, business/finance and science technology is (< 12k)
1. Politics, non-political and askindia dominate with more than 20k posts each
1. Those with less data points need to be further investigated to find if they can be classified easily.
1. Some like coronavirus and caa-nrc-npr were boosted by scraping more recent data, but caa-nrc-npr still remained meagre

### Find number of words in text columns
1. Find the avg, min and max number of words for title, selftext, comment1, comment2 and comment3.
1. Title has max 62 words, with comments going in thousands and selftext 4k
1. Averages are 11 for title, 24 for selftext, and (35, 19, 13) for the comments
1. While this information is useful, frequency line plots will make it more clear

### Frequency plot of no of words in title
1. Find the frequency distribution of no of words in title
1. Gain insight into how much signal to take from title
1. Peak at ~11, only 350 of 126k greater than 50 in length
1. 121k of 126k less than 25 words in length
1. Can use less number of words as most of the important information is in the beginning of the titles.
1. What I got from this analysis is while processing 60 tokens form title is possible, why take so much input when only few posts contain that much data

### Frequency plot of no of words in selftext 
1. Find frequency distribution of no of words in selftext
1. Peaks at 24, maximum do not exceed 200 words
1. Also a lot is empty (102k)
1. Adding selftext may help in the rest of 24k cases)
1. Zero values were removed as this would just generate a plot with a impulse at 0 due to their huge number

### Frequency plots of no of words in comments
1. Find frequency distribution of comment size
1. Most are zero beyond 300
1. While nonempty for comments were less than selftext, the graph is more heavily skewed to the left because avg length of comments is less than selftext
1. Even if comments are added, most come out at very tiny lengths
1. Zero values were removed as this would just generate a plot with a impulse at 0 due to their huge number

### Frequency plot of no of words in selftext + title 
1. Find frequency distribution of selftext + title
1. Now zero values need not be removed as title has minimum one word
1. The peak is still sub 25, adding selftext to title still does not skew the distribution much
1. Considering 102k of the selftext entries are empty, this does not surprise much
1. All this analysis still ignores the flairs, so further analysis tries to get information specific to flairs

### Variation of title length with flair
1. Mean and standard deviation of no of words in title were calculated separately for each flair
1. caa-nrc-npr, coronavirus, politics and policy/economy have the most number of words in title
1. business/finance, scheduled, photography and food have the least number of words
1. Number of words in scheduled is too little, combined with the fact that amount of data is meagre (<1000), so I went ahead looking for some structure in the data for flair scheduled

### Investigating flair scheduled
1. There is structure in r/india scheduled flair, they have the words biweekly, weekly, random daily, late night random etc.
1. There are are some outliers, form 2017, these are little in number ~30, and as they still exist (I visited the links), I decided not to remove them
1. The posts are less noisy the more recent they are, this combined with the structure should make the classifier's job easy.

### Variation of selftext length with flair
1. Mean and standard deviation of no of words in selftext were calculated separately for each flair
1. Scheduled, food and photography get boosted
1. Zero values were removed as then all would skew towards zero, but this made the plot over optimistic, so it was combined with title

### Variation of selftext + title length with flair
1. Mean and standard deviation of no of words in selftext + title were calculated separately for each flair
1. The numbers increase, but not too much like the previous biased plot
1. Need to see both on same plot to find out if there is any benefit of adding selftext to title

### Variation of title with flair and variation of selftext + title with flair on same plot
1. Mean and standard deviations for both were plotted, another one without std was plotted to exaggerate nearness of points
1. While some flairs do get boosted a lot, they are those flairs which were okay at the start
1. caa-nrc-npr has little number of data points, and adding selftext does not increase the numbers much
1. photography also does not get a boost
1. Business/Finance increases, but it already has large number of data points
1. Sports get boosted a little, but it is okay considering the title length only
1. Food gets a significant boost, and it can benefit from adding selftext

### Investigating flair caa-rnc-npr
1. As data for this flair is meagre, datapoints were seen directly
1. Similar structure to scheduled can be seen, but the words are much more diverse
1. If the classifier is able to pick these, this class should not suffer

