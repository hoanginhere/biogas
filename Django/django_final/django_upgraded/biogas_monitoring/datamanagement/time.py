# time_script.py
import time
import subprocess

start_time = time.time()
subprocess.run(['python', 'paperAlgorithm.py'])
end_time = time.time()

print(f"Thời gian thực thi: {end_time - start_time} giây")