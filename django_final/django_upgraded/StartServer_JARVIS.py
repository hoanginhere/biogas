import os
from threading import Thread
import pathlib
x = str(pathlib.Path().resolve())
import time
import webbrowser


def func1():
    os.system('cd ' + x +'\\biogas_monitoring && python manage.py runserver')

def func2():
    os.system('cd ' + x +'\\biogas_monitoring\\datamanagement && python mqtt.py')

def func3():
    webbrowser.get("C:/Program Files/Google/Chrome/Application/chrome.exe %s").open("http://127.0.0.1:8000")

if __name__ == '__main__':
    print('JARVIS: Hello sir, Initiating server')
    Thread(target = func1).start()
    time.sleep(5)
    Thread(target = func2).start()
    Thread(target = func3).start()


