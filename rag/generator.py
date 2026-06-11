import requests
import time


def generate_answer(query: str, chunks: list[dict], api_key: str, retries: int = 3) -> dict:
    context = "\n\n".join(
        [f"[Page {c['page']}]: {c['text']}" for c in chunks]
    )

    prompt = f"""You are a helpful document assistant. Answer the user's question using ONLY the context provided below. Be concise and precise.

Rules:
- Use ONLY the context below to answer.
- If the answer is not in the context, say: "I couldn't find this in the document."
- Always end your answer with a line: "Sources: Page X, Page Y" listing the page numbers you used.

Context:
{context}

Question: {query}

Answer:"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 1024,
            "temperature": 0.2
        }
    }

    for attempt in range(retries):
        response = requests.post(url, json=payload, timeout=30)

        if response.ok:
            data = response.json()
            answer_text = data["candidates"][0]["content"]["parts"][0]["text"]
            source_pages = sorted(set(c["page"] for c in chunks))
            return {
                "answer": answer_text,
                "sources": source_pages,
                "chunks": chunks
            }

        error = response.json().get("error", {})
        error_msg = error.get("message", "Unknown error")

        # If high demand or rate limit — wait and retry
        if response.status_code in [429, 503]:
            wait = 10 * (attempt + 1)   # 10s, 20s, 30s
            print(f"Attempt {attempt+1} failed: {error_msg}. Retrying in {wait}s...")
            time.sleep(wait)
            continue

        # Any other error — raise immediately
        raise Exception(f"Gemini API error: {error_msg}")

    raise Exception("Gemini API is currently busy. Please try again in a minute.")
    
