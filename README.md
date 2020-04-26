# Reddit-Flair-Predictor
Predicts flair for posts in r/india

## Reproduce Development Environment
* Tested on python 3.7.3
* Clone the repository and cd into it
* Then execute the following:
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```
* After the install finishes,
```
python3 -c 'import nltk; nltk.download("punkt")'
```

## Folder info
* scraper: scraper files to scrape posts from r/india
* eda: exploratory data analysis on r/india posts 
* flair predictor model: flair predictor model to predict flairs on r/india
* flair predictor model/writeup: analysis of flair predictor model
* webapp: flair predictor webapp
* experiment log: experiment log while developing the solution
* extras: contains other scripts, notes

## File info
* requirements.txt: dependencies
