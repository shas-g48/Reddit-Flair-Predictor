import requests
import json
import time
import datetime
import praw
import nltk
import string
import time

# initialization for praw
from praw.models import MoreComments
reddit = praw.Reddit()
subreddit = 'india'

# gets submissions in a list of dicts
def pushshift_fetch(before, sub):
    url = 'https://api.pushshift.io/reddit/search/submission/?before='+str(before)+'&size=1000'+'&subreddit='+sub
    print("#debug url: ", url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']

# processes data by removing undesirable characters
def process_text(text):
    import re
    # remove urls from text
    text = re.sub(r'https\S+', '', text)
    tokens = nltk.tokenize.word_tokenize(text)
    # remove usernames of form u/username from text
    tokens = [re.sub(r'u\/.*', '', i) for i in tokens]
    # remove subreddit of form r/subreddit from text
    tokens = [re.sub(r'r\/.*', '', i) for i in tokens]
    # fasttext does not need all to be lowercase
    #tokens = [w.lower() for w in tokens]
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    #words = [word for word in stripped if word.isalpha()]
    words = [re.sub(r'[0-9]', '', i) for i in stripped]
    #text = ' '.join(words)
    words = [x for x in words if x!= '']
    text = ' '.join(words)
    return text

# checks if post is still valid currently
def check(sub):
    if sub.author != None and sub.selftext != '[removed]' and sub.removed_by_category == None and sub.removal_reason == None and sub.removed_by == None and sub.report_reasons == None and sub.banned_by == None and sub.banned_at_utc == None:
        return True
    else:
        return False

# Write out data captured to output file
def writeout(posts):
    with open('data2.txt', 'a') as f:
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


prevdate=''
for i in range(300):
    if i==0:
        date = 1583304037
        #date = 1584148366
        #date = int(time.time())
    else:
        date=prevdate
    
    data = pushshift_fetch(date, subreddit)
    total_posts = len(data)
    accepted = 0
    print("#debug total_posts: ", total_posts)
    
    #for i in data[0]:
    #    print(i,end=":")
    #    print(data[0][i])
    
    prevdate = data[-1]['created_utc']
    
    print("#debug prevdate: ", prevdate)
    
    posts = []
    #print("#debug big iter i:", i)
    """for submission in data:
        print(process_text(submission['title']))
        print(process_text(submission['selftext']))
        print(datetime.datetime.fromtimestamp(submission['created_utc']))
        print(submission['id'])
        print(submission['created_utc'])
        print(submission['permalink']) """

    
    #import pprint
    #links = []
    ids = []
    for submission in data:
        # check if original post has a flair at time of submission
        #prevdate = submission['created_utc']
        if 'link_flair_text' in submission.keys():
            #links.append('https://www.reddit.com'+submission['permalink'])
            ids.append('t3_'+submission['id'])
    
    for n in range(10):
        if (n * 100) < len(ids):
            #fetch_submission = reddit.info(url=links[n*100:(n+1)*100])
            fetch_submission = reddit.info(fullnames=ids[n*100:(n+1)*100])
            #print(fetch_submission)
            #print(len(fetch_submission))
            # check if post is valid now
            #sub = reddit.submission(url='https://www.reddit.com'+submission['permalink'])
            
            num = 0        
            for sub in fetch_submission:
                #time.sleep(2)
                #print('ow')
                #print("#debug outside")
                #print(sub.title)
                #print(sub.id)
                #pprint.pprint(vars(sub))
                
                if check(sub):
                    accepted += 1
                    #print("#debug :", sub.selftext)
                    d = dict()
                    #print("#debug inside")
                    #print(sub.title)
                    #print(sub.id)
                    d['title'] = process_text(sub.title)
                    d['flair'] = sub.link_flair_text
                    d['selftext'] = process_text(sub.selftext)
                    d['date'] = datetime.datetime.fromtimestamp(sub.created_utc)
                    d['sub_id'] = sub.id
                    d['date_raw'] = sub.created_utc
                    d['permalink'] = sub.permalink
                    comments = []
                    for top_level_comment in sub.comments:
                        c = ''
                
                        if isinstance(top_level_comment, MoreComments):
                            comments.append(c)
                        else:
                            c += process_text(top_level_comment.body)
                            comments.append(c)
    
                    for _ in range(3-len(comments)):
                        comments.append('')
    
                    d['c1'] = comments[0]
                    d['c2'] = comments[1]
                    d['c3'] = comments[2]
                    posts.append(d)

                    num+=1
                    if num % 10 == 0:
                        print("#debug data processed:", num)
                    if num % 20 == 0:
                        writeout(posts)
                        print('#debug data written out', num)
                        posts = []
            writeout(posts)
            posts=[]
    print("#debug big iter: ", i)
    print("#debug accepted {} of {} posts".format(accepted, total_posts))
    #time.sleep(60*20)


"""for p in posts:
    print('start of post:')
    for i in p:
        print(i, end=":")
        print(p[i])"""
