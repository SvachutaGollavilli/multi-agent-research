# tests/test_tools.py
# Tests for search_web, search_wikipedia, and async_search_all.
# All HTTP calls are mocked -- zero API spend.

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# -----------------------------------------------------------------
# search_web (Tavily)
# -----------------------------------------------------------------

class TestSearchWeb:
    def _make_tavily_response(self, n: int = 3) -> dict:
        return {
            "results": [
                {
                    "title":   f"Result {i}",
                    "url":     f"https://example.com/{i}",
                    "content": f"Content for result {i}" * 5,
                }
                for i in range(n)
            ]
        }

    def test_returns_list(self):
        mock_client = MagicMock()
        mock_client.search.return_value = self._make_tavily_response(3)
        with patch("src.tools.search.TavilyClient", return_value=mock_client):
            from src.tools.search import search_web
            results = search_web("What is FAISS?")
        assert isinstance(results, list)

    def test_returns_correct_count(self):
        mock_client = MagicMock()
        mock_client.search.return_value = self._make_tavily_response(5)
        with patch("src.tools.search.TavilyClient", return_value=mock_client):
            from src.tools.search import search_web
            results = search_web("query", max_results=5)
        assert len(results) == 5

    def test_results_have_required_keys(self):
        mock_client = MagicMock()
        mock_client.search.return_value = self._make_tavily_response(2)
        with patch("src.tools.search.TavilyClient", return_value=mock_client):
            from src.tools.search import search_web
            results = search_web("query")
        for r in results:
            assert "title"   in r
            assert "url"     in r
            assert "content" in r

    def test_max_results_respected(self):
        mock_client = MagicMock()
        mock_client.search.return_value = self._make_tavily_response(3)
        with patch("src.tools.search.TavilyClient", return_value=mock_client):
            from src.tools.search import search_web
            search_web("query", max_results=4)
        _, kwargs = mock_client.search.call_args
        assert kwargs.get("max_results") == 4 or mock_client.search.call_args[0][1] == 4

    def test_empty_response_returns_empty_list(self):
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        with patch("src.tools.search.TavilyClient", return_value=mock_client):
            from src.tools.search import search_web
            assert search_web("query") == []


# -----------------------------------------------------------------
# search_wikipedia
# -----------------------------------------------------------------

class TestSearchWikipedia:
    def _make_wiki_page(self, title: str, url: str, summary: str):
        page = MagicMock()
        page.title   = title
        page.url     = url
        page.summary = summary
        return page

    def test_returns_list(self):
        page = self._make_wiki_page("FAISS", "https://en.wikipedia.org/wiki/FAISS",
                                    "FAISS is a library " * 30)
        with patch("src.tools.wikipedia.wikipedia.search", return_value=["FAISS"]), \
             patch("src.tools.wikipedia.wikipedia.page",   return_value=page):
            from src.tools.wikipedia import search_wikipedia
            results = search_wikipedia("What is FAISS?")
        assert isinstance(results, list)

    def test_results_have_required_keys(self):
        page = self._make_wiki_page("FAISS", "https://en.wikipedia.org/wiki/FAISS",
                                    "FAISS is a library " * 30)
        with patch("src.tools.wikipedia.wikipedia.search", return_value=["FAISS"]), \
             patch("src.tools.wikipedia.wikipedia.page",   return_value=page):
            from src.tools.wikipedia import search_wikipedia
            results = search_wikipedia("FAISS")
        for r in results:
            assert "title"   in r
            assert "url"     in r
            assert "content" in r

    def test_content_is_truncated_to_1000_chars(self):
        long_summary = "word " * 500  # >2000 chars
        page = self._make_wiki_page("FAISS", "https://en.wikipedia.org/wiki/FAISS",
                                    long_summary)
        with patch("src.tools.wikipedia.wikipedia.search", return_value=["FAISS"]), \
             patch("src.tools.wikipedia.wikipedia.page",   return_value=page):
            from src.tools.wikipedia import search_wikipedia
            results = search_wikipedia("FAISS")
        assert len(results[0]["content"]) <= 1000

    def test_disambiguation_tries_first_option(self):
        import wikipedia as wiki_module
        page = self._make_wiki_page("Python (programming)", "https://en.wikipedia.org/wiki/Python",
                                    "Python is a language " * 30)
        with patch("src.tools.wikipedia.wikipedia.search", return_value=["Python"]), \
             patch("src.tools.wikipedia.wikipedia.page",
                   side_effect=[
                       wiki_module.DisambiguationError("Python", ["Python (programming)", "Python (snake)"]),
                       page,
                   ]):
            from src.tools.wikipedia import search_wikipedia
            results = search_wikipedia("Python")
        assert len(results) == 1
        assert results[0]["title"] == "Python (programming)"

    def test_empty_search_results_returns_empty_list(self):
        with patch("src.tools.wikipedia.wikipedia.search", return_value=[]):
            from src.tools.wikipedia import search_wikipedia
            assert search_wikipedia("xyzzy totally unknown") == []

    def test_page_error_is_skipped(self):
        import wikipedia as wiki_module
        with patch("src.tools.wikipedia.wikipedia.search", return_value=["NonexistentPage"]), \
             patch("src.tools.wikipedia.wikipedia.page",   side_effect=wiki_module.PageError("NonexistentPage")):
            from src.tools.wikipedia import search_wikipedia
            results = search_wikipedia("NonexistentPage")
        assert results == []


# -----------------------------------------------------------------
# async_search_web
# -----------------------------------------------------------------

class TestAsyncSearchWeb:
    def test_returns_list(self):
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value={
            "results": [
                {"title": "T1", "url": "https://a.com", "content": "c1"},
                {"title": "T2", "url": "https://b.com", "content": "c2"},
            ]
        })
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__  = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_ctx)
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__  = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            from src.tools.async_search import async_search_web
            results = asyncio.run(async_search_web("FAISS"))

        assert isinstance(results, list)
        assert len(results) == 2

    def test_http_error_returns_empty_list(self):
        import aiohttp
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("timeout"))
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__  = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            from src.tools.async_search import async_search_web
            results = asyncio.run(async_search_web("FAISS"))

        assert results == []


# -----------------------------------------------------------------
# async_search_all (integration: gather both sources)
# -----------------------------------------------------------------

class TestAsyncSearchAll:
    def _mock_searches(self, tavily_results=None, wiki_results=None):
        return (
            patch("src.tools.async_search.async_search_web",
                  new=AsyncMock(return_value=tavily_results or [])),
            patch("src.tools.async_search.async_search_wikipedia",
                  new=AsyncMock(return_value=wiki_results or [])),
        )

    def test_both_tools_merges_results(self):
        tavily = [{"title": "T", "url": "https://tavily.com/a", "content": "c"}]
        wiki   = [{"title": "W", "url": "https://wikipedia.org/wiki/X", "content": "c"}]
        with patch("src.tools.async_search.async_search_web",      new=AsyncMock(return_value=tavily)), \
             patch("src.tools.async_search.async_search_wikipedia", new=AsyncMock(return_value=wiki)):
            from src.tools.async_search import async_search_all
            results = asyncio.run(async_search_all("query", tool="both"))
        assert len(results) == 2

    def test_deduplicates_by_url(self):
        shared = [{"title": "X", "url": "https://shared.com", "content": "c"}]
        with patch("src.tools.async_search.async_search_web",      new=AsyncMock(return_value=shared)), \
             patch("src.tools.async_search.async_search_wikipedia", new=AsyncMock(return_value=shared)):
            from src.tools.async_search import async_search_all
            results = asyncio.run(async_search_all("query", tool="both"))
        assert len(results) == 1  # dupe removed

    def test_tavily_only_skips_wikipedia(self):
        tavily = [{"title": "T", "url": "https://t.com", "content": "c"}]
        wiki_mock = AsyncMock()
        with patch("src.tools.async_search.async_search_web",      new=AsyncMock(return_value=tavily)), \
             patch("src.tools.async_search.async_search_wikipedia", new=wiki_mock):
            from src.tools.async_search import async_search_all
            asyncio.run(async_search_all("query", tool="tavily"))
        wiki_mock.assert_not_called()

    def test_empty_tool_returns_empty(self):
        from src.tools.async_search import async_search_all
        results = asyncio.run(async_search_all("query", tool="unknown_tool"))
        assert results == []
