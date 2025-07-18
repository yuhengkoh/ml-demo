#checks if log file exists, if not creates it
import os
import sys
import datetime

logfile = "log.txt"
if not os.path.exists(logfile):
    with open(logfile, 'w') as f:
        timenow = str(datetime.datetime.now())
        f.write(timenow + " Log file created.\n")


#hpc_file is the file to be run on the HPC
try:
    import hpc_file #change this to the name of your file
except ImportError as e:
    with open(logfile, 'a') as f:
        f.write("\n")
        timenow = str(datetime.datetime.now())
        outputstr = timenow + " file not found\n"
        f.write(outputstr)
    sys.exit()

try:
    python hpc_file.py # change this to the name of your file
except Exception as e:
    with open(logfile, 'a') as f:
        f.write("\n")
        timenow = str(datetime.datetime.now())
        outputstr = timenow + " error running file: " + str(e) + "\n"
        f.write(outputstr)
    sys.exit()

timenow = str(datetime.datetime.now())
outputstr = timenow + " file run successfully\n"
f.write(outputstr)
'''
folder_name = "verify"
# You can specify a full path like:
# folder_path = "/path/to/your/desired/location/new_folder"
# Or create it in the current working directory:
folder_path = folder_name 

try:
    os.makedirs(folder_path, exist_ok=True) 
    print(f"Folder '{folder_path}' created successfully.")
except OSError as e:
    print(f"Error creating folder '{folder_path}': {e}")
'''