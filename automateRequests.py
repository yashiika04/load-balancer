import requests
import concurrent.futures
import time
import random

def make_request(url, delay):
    """Sends a request to the URL with an optional delay."""
    time.sleep(delay)  
    try:
        response = requests.get(url)
        return response.status_code, response.text
    except requests.RequestException as e:
        return None, str(e)

def send_requests(url, num_requests):
    """Sends multiple requests with a slight delay between them."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_request = {
            executor.submit(make_request, url, random.uniform(0.5, 1)): i   
            for i in range(num_requests)
        }
        for future in concurrent.futures.as_completed(future_to_request):
            status, text = future.result()
            results.append((status, text))
    return results

if __name__ == "__main__":

    url = "http://localhost:8005/heavy-task"   
    responses = send_requests(url, num_requests=30)
    
    for i, (status, text) in enumerate(responses):
        print(f"Request {i+1}: Status {status}, Response: {text[:100]}")
