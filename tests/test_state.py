from src.models.state import ResearchState


def test_state_requires_query():
    """State must always have a query"""
    try:
        s = ResearchState()  # no query — should fail
        assert False, "Should have raised an error"
    except Exception:
        pass


def test_state_default_values():
    """search_results defaults to empty list, report and error to None"""
    s = ResearchState(query="test question")
    assert s.search_results == []
    assert s.report is None
    assert s.error is None


def test_state_accepts_search_results():
    """State should accept a list of result dicts"""
    results = [{"title": "Test", "url": "http://test.com", "content": "Test content"}]
    s = ResearchState(query="test", search_results=results)
    assert len(s.search_results) == 1
    assert s.search_results[0]["title"] == "Test"


def test_state_accepts_error():
    """State should be able to store an error message"""
    s = ResearchState(query="test", error="Something went wrong")
    assert s.error == "Something went wrong"
