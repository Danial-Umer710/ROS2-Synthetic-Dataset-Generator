import math
import time
import requests
import subprocess

def get_windows_host_ip():
    try:
        cmd = "ip route show | awk '/default/ {print $3}'"
        ip = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        if ip:
            return ip
    except Exception as e:
        pass
    return "127.0.0.1"

# PORT INCREMENTED TO 12348 TO BYPASS ZOMBIES
PORT = 12348
HOST_IP = get_windows_host_ip()
URL = f"http://{HOST_IP}:{PORT}"
TOTAL_FRAMES = 1500

print(f"Starting Extreme Domain-Randomized Sweep against Windows Host: {URL}")

for i in range(1203,TOTAL_FRAMES):
    angle = (i / TOTAL_FRAMES) * (2 * math.pi)

    payload = {
        "frame_index": i,
        "orbit_angle": angle,
        "orbit_radius": 0.15,
        "orbit_height": 0.08,
        "randomize": True 
    }
    
    try:
        response = requests.post(URL, json=payload, timeout=120)
        if response.status_code == 200:
            print(f"Successfully rendered frame {i:04d} from orbital angle {math.degrees(angle):.1f}°")
        else:
            print(f"Server error on frame {i:04d}: Status {response.status_code}")
    except Exception as e:
        print(f"Request failed for frame {i:04d}: {e}")
        break

    time.sleep(0.1)

print("Dataset sweep complete!")
