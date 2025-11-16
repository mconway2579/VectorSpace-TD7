# run_all_encoders.py
import subprocess
import sys

commands = [
    [sys.executable, "main.py", "--encoder", "nflow",    "--max_timesteps", "2000000"],
    [sys.executable, "main.py", "--encoder", "addition", "--max_timesteps", "2000000"],
    [sys.executable, "main.py", "--encoder", "td7",      "--max_timesteps", "2000000"],
]

for cmd in commands:
    print("\n==================================================")
    print("Running:", " ".join(cmd))
    print("==================================================")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}: {' '.join(cmd)}")
        break
