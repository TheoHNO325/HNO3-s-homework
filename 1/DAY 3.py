# DAY 3

a = 1
for i in range(9):
    a = (a+1)*2
print(a)

for i in range(1, 8, 2): 
    print(('*' * i).center(7))
for i in range(5, 0, -2):  
    print(('*' * i).center(7))

num = input("第三题输入")
digits = len(num)
print(f"这是{digits}位数")
for i in range(len(num)-1, -1, -1):
    print(num[i], end=" ")
print()

num = input("第四题输入")
if num == num[::-1]:
    print("是")
else:
    print("不是")

year = int(input("第五题输入"))
if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    print("是")
else:
    print("不是")

year,month,day = map(int,(input("第六题输入年月日：")).split())
month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    month_days[1] = 29
total_days = 0
for i in range(month - 1):
    total_days += month_days[i]
total_days += day
print(f"第{total_days}天")
