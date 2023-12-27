import time
import psutil
import os


def measure_gpu_load_os():
    try:
        output = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader").read()
        # gpu_util = float(output)
    except FileNotFoundError:
        print("Warning: nvidia-smi not found. GPU load measurement unavailable.")
        output = None
    
    return output


# Enhanced function for multiple stats:

def measure_gpu_stats():
    """Measures GPU utilization, memory usage, and temperature."""

    nvidia_smi = "nvidia-smi"  # Path to nvidia-smi executable

    try:
        # Get GPU stats using nvidia-smi
        process = psutil.Popen([nvidia-smi, "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader"], stdout=process.PIPE)
        output, _ = process.communicate()
        gpu_stats = output.decode().splitlines()[0].split(",")  # Extract stats into a list
        gpu_util = float(gpu_stats[0])  # GPU utilization
        gpu_memory_used = int(gpu_stats[1])  # Memory usage in MiB
        gpu_temp = int(gpu_stats[2])  # Temperature in Celsius

    except (FileNotFoundError, psutil.CalledProcessError):
        print("Warning: nvidia-smi not found or failed to execute. GPU stats unavailable.")
        gpu_util = None
        gpu_memory_used = None
        gpu_temp = None

    return gpu_util, gpu_memory_used, gpu_temp


# Example usage:
while True:
    gpu_load = measure_gpu_load_os()
    if gpu_load is not None:
        # print("GPU load:", gpu_load, "%")
        print(gpu_load)
    else:
        print("GPU load measurement unavailable.")
    
    time.sleep(1)  # Adjust monitoring interval as needed

# ... further functions and usage ...
