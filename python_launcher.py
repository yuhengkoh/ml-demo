'''
HOW TO USE:
1. Change hpc_file to the name of your Python file (without the .py extension) that you want to run on the HPC.
2. Use launcher.sh to run this script on the HPC
3. Use qsub on launcher.sh
'''

#checks if log file exists, if not creates it
import os
import sys
import datetime

logfile = "log.txt"
if not os.path.exists(logfile): #makes log file if it does not exist
    with open(logfile, 'w') as f:
        timenow = str(datetime.datetime.now())
        f.write(timenow + " Log file created.\n")
else: #else adds new line
    with open(logfile, 'a') as f:
        timenow = str(datetime.datetime.now())
        f.write("\n")

#hpc_file is the file to be run on the HPC, without the .py extension
try:
    import hpc_file #change this to the name of your file
except Exception as e:
    with open(logfile, 'a') as f:
        timenow = str(datetime.datetime.now())
        outputstr = timenow + " error running file: " + str(e) + "\n"
        f.write(outputstr)
    sys.exit()

with open(logfile, 'a') as f:
    timenow = str(datetime.datetime.now())
    outputstr = timenow + " file run successfully\n"
    f.write(outputstr)
