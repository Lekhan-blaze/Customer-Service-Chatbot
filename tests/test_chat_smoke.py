"""Smoke test for the existing chat flow in langchain_helper with mocked LLM (no API key required)."""

from langchain_core.language_models.fake import FakeListLLM
import langchain_helper


def test_existing_chat_flow_returns_answer(monkeypatch):
    """Verify get_qa_chain returns expected structure and answer with mocked LLM."""
    mock_llm = FakeListLLM(responses=["Yes, you can take this course at your own pace."])
    monkeypatch.setattr(langchain_helper, "llm", mock_llm)

    chain = langchain_helper.get_qa_chain()
    response = chain.invoke({"query": "Can I take this course?"})

    assert isinstance(response, dict)
    assert "result" in response
    assert response["result"] == "Yes, you can take this course at your own pace."
    assert "source_documents" in response
    assert len(response["source_documents"]) > 0


def test_existing_chat_flow_fallback(monkeypatch):
    """Verify fallback response behavior when query is outside domain context."""
    mock_llm = FakeListLLM(responses=["I don't know."])
    monkeypatch.setattr(langchain_helper, "llm", mock_llm)

    chain = langchain_helper.get_qa_chain()
    response = chain.invoke({"query": "What is quantum gravity?"})

    assert response["result"] == "I don't know."
