import subprocess
import psutil
import platform

def get_system_info():
    cpu_name, cpu_max_memory = get_cpu_info()
    gpu_name, gpu_max_memory = get_gpu_info()

    return cpu_name, cpu_max_memory, gpu_name, gpu_max_memory

def get_cpu_info():
    os_name = platform.system()
    cpu_name = ""
    if os_name == "Windows":
        wmic_output = subprocess.check_output("wmic cpu get name", shell=True).decode()
        cpu_name = wmic_output.strip().split('\n')[1].strip()

    elif os_name == "Linux":
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if "model name" in line:
                    cpu_name= line.split(':')[1].strip()

    elif os_name == "Darwin":  # macOS is identified as Darwin
        sysctl_output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode()
        cpu_name= sysctl_output.strip()

    cpu_name = remove_word_from_string(cpu_name, "Processor")

    cpu_memory_bytes = psutil.virtual_memory().total
    cpu_max_memory = cpu_memory_bytes / (1024 ** 3)

    return cpu_name, cpu_max_memory

def get_cpu_usage():
    memory_usage_bytes = psutil.virtual_memory().used
    cpu_memory_used = memory_usage_bytes / (1024 ** 3)
    cpu_utilization = psutil.cpu_percent(interval=1)

    return cpu_utilization, cpu_memory_used
    
def get_gpu_info():
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip().split('\n')
    gpu_stats = [line.split(', ') for line in output]
    gpu_name = gpu_stats[0][0]
    gpu_max_memory = int(gpu_stats[0][1]) / 1024

    return gpu_name, gpu_max_memory

def get_gpu_usage():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8').strip().split('\n')
        gpu_stats = [line.split(', ') for line in output]
        gpu_utilization = float(gpu_stats[0][0])
        gpu_memory_used = int(gpu_stats[0][1]) / 1024

        return gpu_utilization, gpu_memory_used
    except Exception as e:
        print("Error accessing GPU stats:", e)
        return 0.0, 0.0, 0.0
    
def remove_word_from_string(input_string, word_to_remove):
    words = input_string.split()
    filtered_words = [word for word in words if word != word_to_remove]
    return ' '.join(filtered_words)