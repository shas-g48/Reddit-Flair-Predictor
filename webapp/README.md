# Webapp to predict flairs on r/india

## Run
* cd into the directory, then 

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
