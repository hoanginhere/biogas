import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("192.168.1.205",8001))
server.listen()
while(True):
    client,addr =server.accept()
    file_name=client.recv(20).decode()
    # file_size = client.recv(20).decode()
    file=open('biogas_monitoring/data_vibration/'+file_name,"wb")
    file_bytes=b""
    done = False
    while not done:
        data = client.recv(1024)
        if file_bytes[-5:] == b"<END>":
            file_bytes += data
            done= True
        else:
            file_bytes += data
        
    file.write(file_bytes)
    file.seek(-5,2)
    file.truncate()
    file.close()
# client.close()
# server.close()