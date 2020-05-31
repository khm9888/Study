#module q2

def calcSpaces(str1):
    count = 0
    for everyLetter in str1:
        if everyLetter == " ":
            count+=1
    return count

def calcDigits(str1):
    count = 0
    for everyLetter in str1:
        if everyLetter.isdigit():
            count+=1
    return count

def calcLetters(str1):
    count = 0
    for everyLetter in str1:
        if everyLetter.isalpha():
            count+=1
    return count
    
