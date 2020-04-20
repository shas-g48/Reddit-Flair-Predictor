import time
import datetime
import requests
import json
import string
import re

import nltk
import praw
from praw.models import MoreComments

# get a reddit instance and set subreddit
reddit = praw.Reddit(client_id='WI6HW6PU0KWLtQ',
                     client_secret='URXonv9_7VKTgARm3V-tAXH_Jco',
                     user_agent='indiasubscrape')
subreddit = 'india'

def pushshift_fetch(before, sub):
    """
    get 1k submissions before a date from a subreddit
    before: date before which to get 1k posts in unix format
    sub: subreddit
    """
    # construct query url and print it
    url = 'https://api.pushshift.io/reddit/search/submission/?before='\
        + str(before) + '&size=1000' + '&subreddit=' + sub
    print("#debug url: ", url)

    # get json response
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']

def process_text(text):
    """
    process data by removing undesirable characters
    text: string to be processed
    """
    # remove urls from text and tokenize it
    text = re.sub(r'https\S+', '', text)
    tokens = nltk.tokenize.word_tokenize(text)

    # remove reddit username and subreddit mentions
    tokens = [re.sub(r'u\/.*', '', i) for i in tokens]
    tokens = [re.sub(r'r\/.*', '', i) for i in tokens]

    # remove punctuation
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # remove numeric characters from words
    words = [re.sub(r'[0-9]', '', i) for i in stripped]
    # remove empty tokens and join text
    words = [x for x in words if x != '']
    text = ' '.join(words)

    return text

# checks if post is still valid currently
def check(sub):
    """
    checks if post is valid currently
    sub: praw submission
    """
    if (sub.author != None and sub.selftext != '[removed]' and
        sub.removed_by_category == None and
        sub.removal_reason == None and sub.removed_by == None and
        sub.report_reasons == None and sub.banned_by == None and
        sub.banned_at_utc == None):
        return True
    else:
        return False

def writeout(posts):
    """
    Writeout data captured to output file
    posts: list of dicts, each dict containing data of a post
    data format is (\n separated): flair, title, selftext,
                                   date, comment1, comment2,
                                   comment3, submission id,
                                   date (unix), submission link
    """
    # open in append mode to continue if script interrupted
    with open('data.txt', 'a') as f:
        for i in posts:
            f.write(str(i['flair']))
            f.write('\n')
            f.write(str(i['title']))
            f.write('\n')
            f.write(str(i['selftext']))
            f.write('\n')
            f.write(str(i['date']))
            f.write('\n')
            f.write(str(i['c1']))
            f.write('\n')
            f.write(str(i['c2']))
            f.write('\n')
            f.write(str(i['c3']))
            f.write('\n')
            f.write(str(i['sub_id']))
            f.write('\n')
            f.write(str(i['date_raw']))
            f.write('\n')
            f.write(str('https://www.reddit.com'+i['permalink']))
            f.write('\n')

# variable to hold previous date
prevdate = ''

# start scraping data, every iteration
# gets 1k submissions on an average
# ~300 are valid, which are saved
for i in range(300):

    # initially, get desired date
    # then get date from last iter
    if i == 0:
        date = int(time.time())
    else:
        date = prevdate

    # get 1k posts after date
    data = pushshift_fetch(date, subreddit)
    total_posts = len(data)
    print("#debug total_posts: ", total_posts)

    # get date for next iter
    accepted = 0
    prevdate = data[-1]['created_utc']
    print("#debug prevdate: ", prevdate)

    # hold ids in praw fullname format t3_{id}
    ids = []
    for submission in data:
        # check if original post has a flair at time of submission
        if 'link_flair_text' in submission.keys():
            ids.append('t3_'+submission['id'])

    posts = [] # hold post dicts

    # go over submissions, 100 at a time
    # due to limitation of praw.Reddit.info()
    for n in range(10):

        # check if posts remaining
        if (n * 100) < len(ids):

            # fetch 100 submissions as a batch
            fetch_submission = reddit.info(fullnames=ids[n*100:(n+1)*100])

            # used to give periodic updates
            num = 0

            # process each submission
            for sub in fetch_submission:

                # check validity to accept post
                if check(sub):
                    accepted += 1

                    # hold data for current submission
                    d = dict()

                    # map fetched attributes to desired attributes
                    d['title'] = process_text(sub.title)
                    d['flair'] = sub.link_flair_text
                    d['selftext'] = process_text(sub.selftext)
                    d['date'] = datetime.datetime.fromtimestamp(sub.created_utc)
                    d['sub_id'] = sub.id
                    d['date_raw'] = sub.created_utc
                    d['permalink'] = sub.permalink

                    # hold 3 top level comments
                    comments = []
                    for top_level_comment in sub.comments:
                        # check if comment is invalid
                        # like 'load more comments'
                        # or 'continue this thread' links
                        if isinstance(top_level_comment, MoreComments):
                            comments.append('')
                        else:
                            comments.append(process_text(top_level_comment.body))

                    # fill remaining comments with empty strings
                    for _ in range(3-len(comments)):
                        comments.append('')

                    # add comments to dict
                    d['c1'] = comments[0]
                    d['c2'] = comments[1]
                    d['c3'] = comments[2]

                    # add the data to posts
                    posts.append(d)
                    num += 1

                    # periodically print out progress
                    # and writeout data
                    if num % 10 == 0:
                        print("#debug data processed:", num)
                    if num % 20 == 0:
                        writeout(posts)
                        print('#debug data written out', num)
                        posts = []

            # writeout any remaining posts
            writeout(posts)
            posts=[]

    # give update on total progress
    print("#debug big iter: ", i)
    print("#debug accepted {} of {} posts".format(accepted, total_posts))
