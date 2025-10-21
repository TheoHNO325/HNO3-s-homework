# DAY 4

lst = [1,2,3,4,5]
result = ','.join(str(x) for x in lst)
print(result)

arr = [1,3,5,7,9]
num = 6
for i in range(len(arr)):
    if num < arr[i]:
        arr.insert(i, num)
        break
else:
    arr.append(num)
print(arr)

X = [[12,7,3],
     [4,5,6],
     [7,8,9]]

Y = [[5,8,1],
     [6,7,3],
     [4,5,9]]
result = [[0,0,0],
          [0,0,0],
          [0,0,0]]
for i in range(3):
    for j in range(3):
        result[i][j] = X[i][j] + Y[i][j]
for row in result:
    print(row)

try:
    dict = {
    (1,1):0,
    (2,2):1,
    (3,3):2
    }
    dict = {
    [1,1]:0,
    [2,2]:1,
    [3,3]:2
    }

    dict = {
    {1,1}:0,
    {2,2}:1,
    {3,3}:2
    }

    print(dict ) 
    print(dict ) 
    print(dict )
except:
    print("不可变类型才能作为字典的键,例如元组，数字和字符串。而列表和集合是可变的")

lis = list(range(1000))
remove = [] # 修改：用列表记录要删的索引
for idx in range(len(lis)):
    if lis[idx] % 2 == 1:
        # list.pop(idx) pop之后索引错位
        remove.append(idx)
lis = [lis[i] for i in range(len(lis)) if i not in remove]  # 修改：重建列表
print(lis)


people = list(range(1, 234))
index = 0
while len(people) > 1:
    index = (index + 2) % len(people)
    people.pop(index)
print(people[0])