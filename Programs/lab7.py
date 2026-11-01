#7.Non-Prime numbers
n = int(input("Enter the numbers:"))

count = 0
for i in range(1,n+1):
    if n%i==0:
        count = count+1
if count!=2:
    print(n,"is Not a Prime number.")
else:
    print(n,"is  a prime number")