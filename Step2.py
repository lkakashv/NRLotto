# Step 2

# Simply Just downlloading my file from github repository
import urllib.request

filedata = urllib.request.urlopen('https://lkakashv.github.io/germanlotto.csv')
datatowrite = filedata.read()
 
with open('germanlotto.csv', 'wb') as f:
    f.write(datatowrite)
