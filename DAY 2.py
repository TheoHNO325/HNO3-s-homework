# DAY 2

zifu = input()
alpha = blank = number = 0
for i in zifu:
    if i.isalpha():
        alpha += 1
    elif i.isspace():
        blank += 1
    elif i.isdigit():
        number += 1
print(alpha, blank, number)

total = 0
c = ""
a = input("a")
n = input("n")
for i in range(n):
    current_term += a 
    total += int(c)
print(total)

height = 100 
l = 0  
n = 10  
for i in range(1, n + 1):
    if i == 1:
        l += height
    else:
        l += 2 * height
    height /= 2  
    
print(f"共经过：{l:.2f}米")
print(f"第10次反弹高度：{height:.2f}米")

for num in range(100, 1000):
    hundreds = num // 100
    tens = (num // 10) % 10
    units = num % 10
    
    sum = hundreds**3 + tens**3 + units**3
    
    if sum == num:
        print(num)



primes = []
for num in range(101, 201):
    is_prime = True
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            is_prime = False
            break
    if is_prime:
        primes.append(num)
print(primes)