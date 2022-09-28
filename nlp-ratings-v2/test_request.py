import requests

inference_request = {
    "parameters": {
        "content_type": "pd"
    },
    "inputs": [
        {
          "name": "review",
          "shape": [1, 1],
          "datatype": "BYTES",
          "data": ["_product is excellent! I love it, it's great!"]
        }
    ]
}

endpoint = "http://localhost:8080/v2/models/nlp-ratings-v2/infer"
response = requests.post(endpoint, json=inference_request)

print(response.json())

