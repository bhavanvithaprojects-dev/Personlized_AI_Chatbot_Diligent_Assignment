import subprocess

def generate_response(prompt: str):
    result = subprocess.run(
        ["ollama", "run", "llama2"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()
