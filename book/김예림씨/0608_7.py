#7

chk = True

try:
    text = list(map(int,input("Type the input string:").split(":")))
    for i in range(len(text)):
        if type(text[i])!=int:
            chk=False
            break
        if 0<=text[i]<=255:
            pass
        else:
            chk=False
except:
    chk=False
 
if chk:
    print("Yes, this is an IPV4 address.")
    
else:
    print("No, this is not a valid IPV4 address.")
