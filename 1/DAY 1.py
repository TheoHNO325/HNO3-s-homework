# DAY 1

all = 0
for i in range(1,5):
    for j in range(1,5):
        for k in range(1,5):
            if i != j and j != k and i != k: all += 1

print(all)


x,y,z = input().split()
if x > y:
    x,y = y,x
if x > z:
    x,z = z,x
if y > z:
    y,z = z,y
print(x,y,z)

a0 = 1
a1 = 1

for i in range(19):
    a0 = a0+a1
    print(a0)
    a0,a1 = a1,a0


for i in range(1,10):
    for j in range(1,i+1):
        print(f"{j}*{i}={i*j}",end="\t")
    print()


