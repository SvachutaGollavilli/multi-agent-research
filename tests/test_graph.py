from src.agents.graph import build_graph
from src.models.state import ResearchState


def test_graph_builds_successfully():
    """Graph should compile without errors"""
    pipeline = build_graph()
    assert pipeline is not None


def test_graph_runs_end_to_end():
    """Full pipeline should return search results for a real query"""
    pipeline = build_graph()
    result = pipeline.invoke({"query": "What is LangGraph used for?"})

    assert "query" in result
    assert "search_results" in result
    assert len(result["search_results"]) > 0


def test_graph_state_has_no_error():
    """A valid query should not produce an error in state"""
    pipeline = build_graph()
    result = pipeline.invoke({"query": "Python programming best practices"})

    assert result.get("error") is None


def test_graph_results_have_required_keys():
    """Each search result should have title, url, and content"""
    pipeline = build_graph()
    result = pipeline.invoke({"query": "machine learning trends 2026"})

    for r in result["search_results"]:
        assert "title" in r
        assert "url" in r
        assert "content" in r
