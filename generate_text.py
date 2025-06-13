import requests
import json

def generate_text(prompt, model_url="http://localhost:8000/generate", max_tokens=50, temperature=0.7):
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "n": 1,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post(model_url, headers=headers, data=json.dumps(data))
    response.raise_for_status()  # Raise an exception for bad HTTP status codes
    return response.json()

if __name__ == "__main__":
    prompt = "In a galaxy far, far away,"
    try:
        result = generate_text(prompt)
        print(f"Prompt: {prompt}")
        # Handle potentially varying response structures from vLLM
        # Assuming response is {'text': [[<generated_text>]]} or similar
        if 'text' in result and len(result['text']) > 0 and len(result['text'][0]) > 0:
            print(f"Generated Text: {result['text'][0][0]}")
        elif 'outputs' in result and len(result['outputs']) > 0: # Some vLLM versions might use 'outputs'
            print(f"Generated Text: {result['outputs'][0]['text']}")
        else:
            print("Error: Unexpected response structure from vLLM server.")
            print(f"Full Response: {result}")
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except KeyError:
        print("Error: Unexpected response format from vLLM server.")
