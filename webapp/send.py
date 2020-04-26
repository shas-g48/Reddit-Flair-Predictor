import requests
files = {'upload_file':open('test.txt', 'rb')}
r = requests.post('https://[url]/automated_testing',files=files)
print(r.json())
