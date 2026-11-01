#2.Sum of natural numbers
n = int(input("Enter the no of natural numbers:"))
sum = 0
for i in range(1,n+1):
    sum = sum+i
print("Sum of natural numbers:",sum)