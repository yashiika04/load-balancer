import requests
import threading
import time

TARGET_SERVERS = [
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
    # "http://127.0.0.1:8002",

]

def send_burst(server, num_requests=100, label=""):
    success = 0
    failed = 0
    for i in range(num_requests):
        try:
            r = requests.get(f"{server}/heavy-task", timeout=5)
            if r.status_code == 200:
                success += 1
            else:
                failed += 1
        except Exception:
            failed += 1
    print(f"[{label}] Done — success: {success}, failed: {failed}")

threads = []
for server in TARGET_SERVERS:
    label = server.split(":")[-1]  # port number
    t = threading.Thread(target=send_burst, args=(server, 200, label))
    threads.append(t)

print("Sending burst traffic to 8000 and 8001...")
for t in threads:
    t.start()
for t in threads:
    t.join()

print("Burst complete.")