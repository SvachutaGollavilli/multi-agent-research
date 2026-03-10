# src/tools/wikipedia.py

import logging
import wikipedia

logger = logging.getLogger(__name__)


def search_wikipedia(query: str, max_results: int = 3) -> list[dict]:
    """
    Search Wikipedia and return clean results in the same
    {title, url, content} shape as search_web() — so the rest of
    the pipeline treats both sources identically.

    Uses two-step approach:
      1. wikipedia.search()  → returns list of matching page titles
      2. wikipedia.page()    → fetches full page for each title
    Falls back gracefully on disambiguation or missing pages.
    """
    results = []

    try:
        titles = wikipedia.search(query, results=max_results)
        logger.debug(f"[wikipedia] found {len(titles)} candidate titles")

        for title in titles[:max_results]:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                results.append({
                    "title":   page.title,
                    "url":     page.url,
                    # First 1000 chars is enough context for the analyst
                    "content": page.summary[:1000],
                })
            except wikipedia.DisambiguationError as e:
                # e.options gives alternative titles — try the first one
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    results.append({
                        "title":   page.title,
                        "url":     page.url,
                        "content": page.summary[:1000],
                    })
                except Exception:
                    logger.debug(f"[wikipedia] skipping ambiguous title: {title}")
            except wikipedia.PageError:
                logger.debug(f"[wikipedia] page not found: {title}")
            except Exception as e:
                logger.debug(f"[wikipedia] error fetching '{title}': {e}")

    except Exception as e:
        logger.error(f"[wikipedia] search failed: {e}")

    logger.info(f"[wikipedia] returning {len(results)} results for '{query[:50]}'")
    return results


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    results = search_wikipedia(input("Wikipedia query: "), max_results=3)
    for r in results:
        print(f"\n{r['title']}")
        print(f"{r['url']}")
        print(f"{r['content'][:200]}...")
