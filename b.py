
import praw
import pprint
reddit = praw.Reddit()

#assume you have a Reddit instance bound to variable `reddit`
submission = reddit.submission(url='https://www.reddit.com/r/india/comments/fvz69t/mom_came_up_with_this_impressive_idea_to/')
print(submission.title) # to make it non-lazy
pprint.pprint(vars(submission))
