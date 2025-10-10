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
    import traceback
    # Get the full traceback information, including line numbers
    full_traceback = traceback.format_exc()
    print("Full Traceback:")
    print(full_traceback)

    # Extract specific information about the error
    exc_type, exc_value, exc_traceback = sys.exc_info()
    
    # Get the last frame in the traceback, which typically corresponds to the error line
    tb_list = traceback.extract_tb(exc_traceback)
    last_frame = tb_list[-1]
    
    line_number = last_frame.lineno
    filename = last_frame.filename
    function_name = last_frame.name
    line_of_code = last_frame.line

    print(f"\nError occurred in file: {filename}")
    print(f"Function: {function_name}")
    print(f"Line number: {line_number}")
    print(f"Line of code: {line_of_code}")
    with open(logfile, 'a') as f:
        timenow = str(datetime.datetime.now())
        outputstr = timenow + " error running file: " + str(e) + "\n"
        outputstr += f"Error occurred in file: {filename}\n"
        outputstr += f"Function: {function_name}\n"
        outputstr += f"Line number: {line_number}\n"
        outputstr += f"Line of code: {line_of_code}\n"
        f.write(outputstr)
    sys.exit()

with open(logfile, 'a') as f:
    timenow = str(datetime.datetime.now())
    outputstr = timenow + " file run successfully\n"
    f.write(outputstr)
