import glob
import os
import subprocess
import gc
import time
import torch

def main():
    # Define the directory where your YAML config files are stored
    config_dir = "configs"
    
    # Use glob to list all .yaml files in the directory
    config_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    
    if not config_files:
        print(f"No YAML config files found in directory: {config_dir}")
        return
    
    # Iterate over each config file and call main.py with it
    for config_file in config_files:
        print(f"Running main.py with config file: {config_file}")
        # Run main.py as a subprocess, passing the config file as an argument
        result = subprocess.run(["python", "main.py", config_file])
        
        # Optionally, check the return code to see if execution was successful.
        if result.returncode != 0:
            print(f"Error running main.py with {config_file}. Return code: {result.returncode}")
        
        # Force garbage collection and empty CUDA cache to free GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Optional: sleep a few seconds to allow CUDA memory to fully free up before the next run.
        time.sleep(5)

if __name__ == "__main__":
    main()
