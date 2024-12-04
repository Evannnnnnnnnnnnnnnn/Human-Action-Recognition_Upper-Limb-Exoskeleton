if __name__ == "__main__" :
    print("\033cStarting ...\n") # Clear Terminal

import socket
import time
import sys
import os

try :
    import dotenv
except ModuleNotFoundError :
    sys.exit(f'No module named dotenv try : pip install python-dotenv')


dotenv.load_dotenv()

bufferSize = 1024
try :
    serverPort = int(os.getenv('serverPort_env'))
    serverIP = os.getenv('serverIP_env')
except TypeError :
    sys.exit('\033cPlease open .env.shared and follow instructions')
serverAddress = (serverIP,serverPort)

UDPClient = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

TimeFromClient = str(0)
print(TimeFromClient)
TimeFromClient_bytes = TimeFromClient.encode('utf-8')
UDPClient.sendto(TimeFromClient_bytes, serverAddress)

try :
    Done = False
    print('\033c', end='')
    while not Done :
        messageFromClient = str(input('message to send : '))
        messageFromClient_bytes = messageFromClient.encode('utf-8')

        UDPClient.sendto(messageFromClient_bytes, serverAddress)

        if messageFromClient.lower() == 'done' :
            Done = True
except KeyboardInterrupt :
    UDPClient.sendto('done'.encode('utf-8'), serverAddress)



if __name__ == "__main__" :
    print('\nProgramme Stopped\n')