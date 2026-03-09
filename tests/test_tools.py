from src.tools import search_web

def test_search_returns_results():
    results = search_web("Trending Programming Language")
    assert isinstance(results, list)
    assert len(results)>0

def test_result_required_keys():
    results = search_web("Agentic AI Tutorial")
    for r in results:
        assert 'title' in r
        assert 'url' in r
        assert 'content' in r


def test_search_results_max_respect():
    results = search_web("AI Agent Frameworks", max_results = 4)
    assert len(results) <= 4