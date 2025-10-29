# DAY 5
import random
import string
import os

def san():
    with open('data.txt', 'w') as f:
        for _ in range(100000):
            f.write(str(random.randint(1, 100)) + '\n')

def si():
    os.makedirs('img', exist_ok=True)
    for _ in range(100):
        name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        imgpath = os.path.join('img',f'{name}.png')
        open(imgpath, 'a').close()

def wu():
    img_dir = 'img'
    files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    selected = random.sample(files, 50)
    
    for filename in selected:
        old_path = os.path.join(img_dir, filename)
        new_filename = filename[:-4] + '.jpg'  
        new_path = os.path.join(img_dir, new_filename)
        os.rename(old_path, new_path)

if __name__ == '__main__':
    san()
    si()
    wu()