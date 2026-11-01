#Sum of numbers in a string
s = input("Enter a string:")
sum = 0
for ch in s:
    if ch.isdigit():
     sum = sum + int(ch)
print("The Sum of Numbers in a String is",sum)

