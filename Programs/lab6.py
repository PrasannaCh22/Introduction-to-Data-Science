#6. Prime numbers
n = int(input("Enter the numbers:"))

count = 0
for i in range(1,n+1):
    if n%i==0:
        count = count+1
if count==2:
    print(n,"is a Prime number.")
else:
    print(n,"is not a prime number")