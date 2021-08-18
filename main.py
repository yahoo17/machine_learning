import os
import time

def ee():
    content = '顺德欢迎你'
    while True:
        print(content)
        content = content[1:] + content[0]
        time.sleep(1)
        os.system('clear')


