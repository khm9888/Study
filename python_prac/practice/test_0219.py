import re
 
text = "032-232-3245"
t=input()
regex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
retext = bool(re.match('\d{3}-\d{3}-\d{4}',t))
print(retext)
matchobj = regex.search(text)
phonenumber = matchobj.group()
print(phonenumber) 