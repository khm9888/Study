#PS4 Q1
#Solomon says:
#Ask the user to input any sentence.
#Then call the module solomon_says.py.
#In this module is a function solomon_add()
#that adds the phrase “Solomon says, (user sentence)”
#and returns the new sentence. Print this sentence.
#Example.
#Input: “Give me an apple.”
#Output: “Solomon says, give me an apple.”



# algorithm
# --main code
# take input string
# import module into main code
# call function from module...catch final string returned in a variable
# print that

#--module
#def function...take one string
#add solomon says to that
#return final string.

from module_SolomonSays import solomon_add

text=input()

solomon_add(text)




















#code
#main_SolomonSays.py
import module_SolomonSays
userInput = input("Type sentence")

finalStr = module_SolomonSays.solomon_add(userInput)

print(finalStr)
