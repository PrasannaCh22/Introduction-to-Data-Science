#Fibonacci Series
n = int(input("Enter the range of no's:"))
a = 0
b = 1
for i in range(n):
    print(a,end = " ")
    c = a+b
    a = b 
    b = c