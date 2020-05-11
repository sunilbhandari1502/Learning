# lambda Function definition
sumLambda = lambda x, y: x + y
# Usage of Mambda function
m = 1
n = 2
print("Sum of", m, "and", n, " is", sumLambda(m, n))

# Lambda function example with filter (take a funcition and list as argumetn and retun nback the list element which satify function)
oldlist = [2, 31, 42, 11, 6, 5, 23, 44]
# Usage of lambda function
newlist = list(filter(lambda x: (x % 2 != 0), oldlist))
print(oldlist)
print(newlist)

# Lambda function example with map (take a funcition and list as argumetn and retun back the list element after implementing the function to each element)
oldlist = [2, 31, 42, 11, 6, 5, 23, 44]
# Usage of lambda function
newlist = list(map(lambda x: x + 2, oldlist))
print(oldlist)
print(newlist)

#Recursive Function
def recursion_fact(x):
    # this is a recursive function to find the factorial of an integer
    if x == 1:
        return 1
    else:
        return (x * recursion_fact(x - 1))  # recursive calling


num = 5
print("The factorial of", num, "is", recursion_fact(num))

#Function with more than one return value
def calc(a, b):
    sum = a + b
    diff = a - b
    prod = a * b
    quotient = a / b
    return sum, diff, prod, quotient


a = 10
b = 5
s, d, p, q = calc(a, b)
print("Sum= ", s)
print("Difference= ", d)
print("Product= ", p)
print("Quotient= ", q)
