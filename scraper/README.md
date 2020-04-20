# Scrape posts from r/india

## Run
```
python3 scraper.py
```

## File info
* scraper.py: scraper python script
* scraper.ipynb: scraper jupyter notebook
* data.txt: sample scraped data

## Features
* Get flair, title, selftext, date, 3 top level comments, submission id and link
* Writeout data in batches to resume if interrupted, just need to change date in script
* Filter out some kinds of invalid posts
* Do pre-processing on text

## Some Implementation details
* Uses pushshift api to get ids, then uses praw to get the post
* Gets data with praw in batches of 100 (maximum possible)

## Additional Notes
* If script stalls, set date in script to last successfully scraped, to get last date run:
```
tail -n 10 data.txt
```
* When changing date, use the unix date format without the fraction part
* Instead of removing words like 'english-24', it gets converted to 'english'
