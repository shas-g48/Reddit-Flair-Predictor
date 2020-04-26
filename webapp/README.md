# Webapp to predict flairs on r/india

## Run
* save this file to root of the directory (contains fasttext text embeddings required to run the webapp):
```
https://drive.google.com/open?id=1fyu1nFl8vJnnhh6VnC4BpZE8KG85i0VO
```

* cd into the directory and then run
```
flask run
```

## Usage
* web browser

```
put the link in the text box and submit to get the flair
```

* raw json results using automated testpoint (cli)

```
curl -F 'file=@[file containing reddit links].txt' http://[address shown by flask]/automated_testing
```

* raw json results using python to send request (edit the script to change the filename and url)

```
python3 send.py
```

## Folder info:
* application: flask app

## File info:
* send.py: python script to send request to automated endpoint
