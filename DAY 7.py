import random
import string
import os
import statistics
from pathlib import Path

def first():
    with open('data.csv', 'w') as f:
        for _ in range(10):
            row = [str(random.randint(1, 100)) for _ in range(3)]
            f.write(','.join(row) + '\n')
    
    with open('data.csv', 'r') as f:
        lines = f.readlines()
        el = [int(line.split(',')[1]) for line in lines]
        
    print(max(el))
    print(min(el))
    print(statistics.mean(el))
    print(statistics.median(el))

def second():
    n = int(input("几行"))
    with open('test.txt', 'w') as f:
        for _ in range(n):
            line = ''.join(random.choices(string.printable[:-5], k=random.randint(10, 50)))
            f.write(line + '\n')
    
    with open('test.txt', 'r') as src, open('copy_test.txt', 'w') as dst:
        dst.write(src.read())

def third():
    with open('test.txt', 'r+') as f:
        content = f.read()
        f.seek(0)
        f.write('python' + content + 'python')

def fourth():
    with open('test.txt', 'r') as f1, open('copy_test.txt', 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
    
    diff_lines = []
    for i, (l1, l2) in enumerate(zip(lines1, lines2), 1):
        if l1 != l2:
            diff_lines.append(i)
    
    print(diff_lines)

import os
import random
import string

def fifth():
    num_files = int(input("几个文件: "))
    num_lines = int(input("几个字符: "))
    
    os.makedirs('test', exist_ok=True)
    
    for i in range(num_files):
        file_path = os.path.join('test', f'file_{i}.txt')
        with open(file_path, 'w') as f:
            for _ in range(num_lines):
                line = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
                f.write(line + '\n')
    
    for filename in os.listdir('test'):
        file_path = os.path.join('test', filename)
        
        if os.path.isfile(file_path):
            name, ext = os.path.splitext(filename)
            new_name = name + '-python' + ext
            new_path = os.path.join('test', new_name)
            os.rename(file_path, new_path)
            
            with open(new_path, 'r+') as f:
                lines = f.readlines()
                f.seek(0)  
                for line in lines:
                    f.write(line.strip() + '-python\n')
                f.truncate()  

if __name__ == '__main__':
    first()
    second()
    third()
    fourth()
    fifth()