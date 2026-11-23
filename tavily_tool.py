import os
from pathlib import Path

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

api_key = os.getenv("TAVILY_API_KEY")
client = TavilyClient(api_key=api_key) if api_key else None


def tavily_search(query: str):
    if client is None:
        return "Hotel search unavailable: TAVILY_API_KEY is not configured."

    try:
        response = client.search(query=query, max_results=5)
    except Exception as exc:
        error_name = exc.__class__.__name__
        if error_name == "ForbiddenError":
            return "Hotel search unavailable: your Tavily plan usage limit has been reached. Upgrade the Tavily plan or replace TAVILY_API_KEY with an active key."
        return f"Hotel search unavailable due to a Tavily API error ({error_name})."

    results = []
    for i, r in enumerate(response.get("results", []), start=1):
        title = r.get("title", "Unknown")
        url = r.get("url", "")
        snippet = r.get("content", "").strip()

        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        results.append(f"{i}. **{title}**\nURL: {url}\nSnippet: {snippet}\n")

    return "\n".join(results) if results else "No hotel results found."
