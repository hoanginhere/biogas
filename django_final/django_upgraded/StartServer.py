import os
from threading import Thread
import pathlib
x = str(pathlib.Path().resolve())
import time

print("What IP are we running the server on, Sir?")
server_ip = input()

def func1():
    os.system('cd ' + x +'\\biogas_monitoring && python manage.py runserver'+" "+server_ip)

def func2():
    os.system('cd ' + x +'\\biogas_monitoring\\datamanagement && python mqtt.py')

def func3():
    os.system('python Socket_server.py')

if __name__ == '__main__':
    print('JARVIS: Hello sir, Initiating server')
    Thread(target = func1).start()
    Thread(target = func2).start()
    Thread(target = func3).start()


