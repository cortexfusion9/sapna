import requests
import json

url = "http://127.0.0.1:8000/api/chat"
headers = {
    "Content-Type": "application/json",
    "X-User-Id": "user_plumb_789"
}
payload = {
    "user_query": "What is the total project cost of Apex Tower?"
}

try:
    print(f"Sending POST request to {url}...")
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error during API call: {e}")
