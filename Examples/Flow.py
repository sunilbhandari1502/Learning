# fizzBuzz test
for i in range(1, 20):
    if i % 3 == 0 and i % 5 == 0:
        print("Fizz Buzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)

# Printing a pattern
for i in range(1, 5):
    for j in range(1, i + 1):
        print(i, end=' ')
    print()

# count of particular number in list and use of map for list
str = input("Enter a list(value space separated :")
#map(aFuntion,aSequence) function applies a passed-in function to each item in an iterable object and returns a listi containing all the finction calls results
lis = list(map(int, str.split()))
n = int(input("Enter the number to search for occurrence"))
print(lis)
print("number of occurrence of ", n, "is", lis.count(n), "times")

#count number of character frequency in a given string
inputstr = input("Enter a  string:")
dic = {}
for chr in inputstr:
    if chr in dic:
        dic[chr] += 1
    else:
        dic[chr] = 1
print(dic)
for l, m in dic.items():
    print(l, m)

#takes list of words and returns the length of the longest one

inputlist = input("Enter a  string:")
word_list = inputlist.split()
word_len = []
for n in word_list:
    word_len.append((len(n), n))
    print(word_len)
    word_len.sort()
    print(word_len)
print("longest Word:", word_len[-1][1])
