#!/usr/bin/env python3
import sys
import json
import urllib.request
import urllib.error

DEFAULT_ENDPOINT = "http://localhost:8080/v1/chat/completions"

def check_health(endpoint: str):
    """
    Performs a lightweight ping to the API to ensure the model is responsive.
    We just send a tiny, 1-token prompt to verify the inference engine is alive.
    """
    payload = {
        "model": "mistral",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1
    }
    
    req = urllib.request.Request(
        endpoint, 
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        # A fast timeout for health checks
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                print(json.dumps({"status": "healthy", "service": "mistral_delegate", "endpoint": endpoint}))
                sys.exit(0)
    except Exception as e:
        print(json.dumps({"status": "unhealthy", "service": "mistral_delegate", "error": str(e)}))
        sys.exit(1)

def query_mistral(prompt: str, endpoint: str):
    """
    Executes the actual heavy inference task over the provided plain text/Jinja prompt.
    """
    payload = {
        "model": "mistral",
        "messages": [
            {"role": "system", "content": "You are a specialized data-processing model. Execute the requested task concisely and output ONLY the requested data."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 4096
    }
            
    req = urllib.request.Request(
        endpoint, 
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        # Paling operations might take a while, use a long timeout
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode('utf-8'))
            text_output = result['choices'][0]['message']['content']
            print(json.dumps({"status": "complete", "result": text_output}))
            sys.exit(0)
    except Exception as e:
        print(json.dumps({"status": "failed", "error": str(e)}))
        sys.exit(1)

def main():
    # 1. Ensure zero CLI arguments are passed (enforcing the contract)
    if len(sys.argv) > 1:
        print(json.dumps({"status": "failed", "error": "CLI arguments are strictly forbidden by Paling contract."}))
        sys.exit(1)

    # 2. Check if stdin has data. If not, this is a liveness probe.
    if sys.stdin.isatty():
        check_health(DEFAULT_ENDPOINT)
        return

    # 3. Read JSON payload from stdin
    try:
        raw_input = sys.stdin.read().strip()
        if not raw_input:
            check_health(DEFAULT_ENDPOINT)
            return
            
        payload = json.loads(raw_input)
    except json.JSONDecodeError as e:
        print(json.dumps({"status": "failed", "error": f"Invalid JSON on stdin: {e}"}))
        sys.exit(1)

    action = payload.get("action", "health")
    endpoint = payload.get("endpoint", DEFAULT_ENDPOINT)

    # 4. Route based on requested action
    if action == "health":
        check_health(endpoint)
    elif action == "execute":
        prompt = payload.get("prompt")
        if not prompt:
            print(json.dumps({"status": "failed", "error": "Missing 'prompt' in execution payload."}))
            sys.exit(1)
        
        query_mistral(prompt, endpoint)
    else:
        print(json.dumps({"status": "failed", "error": f"Unknown action: {action}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()
