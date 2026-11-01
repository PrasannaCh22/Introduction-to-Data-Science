#simple Calculator
a = float(input("Enter the first no:"))
b = float(input("Enter the second no:"))
op = input("Enter the operator(+,-,*,/):")
if op == "+":
    print("Result:",a+b)
elif op == "-":
    print("Result:",a-b)
elif op == "*":
    print("Result:",a*b)
elif op == "/":
    print("Result:",a/b)
else:
    print("Invalid Operator.")
