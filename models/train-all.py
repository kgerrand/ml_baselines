'''
This script runs all Python files in subdirectories, and so trains all models.
'''
#-------------------------------------------------------------
import os
import subprocess

root_dir = os.path.dirname(os.path.abspath(__file__))

for subdir, dirs, files in os.walk(root_dir):
    if subdir == root_dir:
        continue
    
    for file in files:
        if file.endswith(".py"):
            script_path = os.path.join(subdir, file)

            file_name = os.path.basename(script_path)
            file_parent = os.path.basename(os.path.dirname(script_path))
            print(f"--- Running {file_parent}/{file_name} ---")
            result = subprocess.run(["python", script_path], capture_output=True, text=True, cwd=subdir)
            
            print("Output:\n", result.stdout)
            if result.stderr:
                print("Errors:\n", result.stderr)