from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

def search_web(query:str, max_results: int = 5) -> list[dict]:
    """
    Search the web using Tavily and return clean results.

    Args:
        query: The search query string
        max_results: How many results to return (default 5), can cap it to a max of 10

    Returns:
        List of dicts with 'title', 'url', 'content' keys 
    """

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    response = client.search(
        query = query,
        max_results = max_results,
        search_depth = 'basic'
    )

    results = []
    for r in response.get('results', []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content":r.get("content","")
        })

    return results


if __name__ == "__main__":
    # Quick manual test - run with: uv run python src/tools/search.py
    results = search_web(str(input("enter the search requirement: ")))
    for r in results:
        print(f"\n {r['title']}")
        print(f"{r['url']}")
        print(f"{r['content'][:200]},,,")