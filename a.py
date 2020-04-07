import praw
import nltk
import string
import datetime
from praw.models import MoreComments
reddit = praw.Reddit()
posts = []

def process_text(text):
    import re
    text = re.sub(r'https\S+', '', text)
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [re.sub(r'u\/.*', '', i) for i in tokens]
    tokens = [re.sub(r'r\/.*', '', i) for i in tokens]
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    text = ' '.join(words)
    return text

num = 0
def writeout(posts):
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

for submission in reddit.subreddit('india').new(limit=None):
    d = dict()
    d['title'] = process_text(submission.title)
    d['flair'] = submission.link_flair_text
    d['selftext'] = process_text(submission.selftext)
    d['date'] = datetime.datetime.fromtimestamp(submission.created)
    d['sub_id'] = submission.id
    d['date_raw'] = submission.created
    #d['date'] = submission.created
    
    comments = []
    
    for top_level_comment in submission.comments:
        c = ''
         
        if isinstance(top_level_comment, MoreComments):
            comments.append(c)
        else:
            c += process_text(top_level_comment.body)
            comments.append(c)
    for i in range(3-len(comments)):
        comments.append('')
    d['c1'] = comments[0]
    d['c2'] = comments[1]
    d['c3'] = comments[2]
    posts.append(d)
    num+=1
    if num % 10 == 0:
        print('#debug data processed:', num)
    if num % 100 == 0:
        writeout(posts)
        print('#debug data written out', num)
        posts = []
    
writeout(posts)
