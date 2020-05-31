#PS4Q8
#With two given lists [1,3,6,78,35,55,57] and
#[12,24,35,24,78,120,155,3], write a program to make a
#list whose elements are the common elements from the two given lists.
#Output:
#[3,78,35]
#(the main part of the code should be inside a function.
#Pass the two lists to a function called commonElements(). )




#ALGORITHM

#declare both lists..list1 list2

#define function...comEle...takes 2 lists
    #create empty list...comList
    #iterate over list1...
        #if element of list1 IN in list2...membership operator in
            #append it to comList
    #return the comList

#call function pass the two lists..catch return from function
#print the common list



















#code
list1 = [1,3,6,78,35,55,57]
list2 = [12,24,35,24,78,120,155,3]

def commonElements(list1, list2):
    comEle = []
    for everyEle in list1:
        if everyEle in list2:
            comEle.append(everyEle)
    return comEle

finalList = commonElements(list1,list2)
print("Common elements are: ", finalList)


