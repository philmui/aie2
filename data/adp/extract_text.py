import json
import argparse

filter_prefix = [
    "Auth Failed",
    "466 Too many requests",
    "404 Not Found",
    "403 Forbidden",
    "Page not found",
    "Get Directions Search",
    "X JavaScript is not available",
    "Saved Items",
    "My Doctor Online Close",
    "Internal server error",
    "Sign in Region Choose",
    "Pinterest Pinterest",
    "Get Care | My Doctor Online Close Internet Explorer not supported",
    "[", "{",
]

filter_text = [
    "Internet Explorer not supported",
    "Page not found",
    "Internal server error",
]

contain_text = [
    "eye",
    "itchy",
    "bump",
    "episcleritis", "Episcleritis",
    "ophthalmology", "Ophthalmology",
]

def extract_text_from_jsonl(jsonl_file):
    with open(jsonl_file, 'r') as f:
        visited = set()
        for line in f:
            data = json.loads(line)
            url = data.get('url', '').strip()
            if url in visited:
                continue
            visited.add(url)
            title = data.get('title', '').strip()
            text = data.get('text', '').strip()
            if any(snippet in text for snippet in filter_text) or \
                    any(text.startswith(prefix) for prefix in filter_prefix) or \
                    len(text) < 32:
                continue
            # elif any(word in text for word in contain_text):
            print(f"{title}. {text}")


parser = argparse.ArgumentParser(description="Extract text from JSONL file")
parser.add_argument("jsonl_file", help="Path to the JSONL file")
if __name__ == "__main__":
    args = parser.parse_args()
    jsonl_file = args.jsonl_file
    if jsonl_file is not None:
        extract_text_from_jsonl(jsonl_file)