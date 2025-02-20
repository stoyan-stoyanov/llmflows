"""Tests for the OpenAI embeddings implementation."""

import pytest
from unittest.mock import Mock, patch
from openai import APIError, Timeout, RateLimitError, APIConnectionError
from llmflows.llms.openai_embeddings import OpenAIEmbeddings
from llmflows.vectorstores.vector_doc import VectorDoc

def test_init_with_valid_api_key():
    """Test initialization with valid API key."""
    embeddings = OpenAIEmbeddings(api_key="test_key")
    assert embeddings.model == "text-embedding-3-small"
    assert embeddings.max_retries == 3
    assert embeddings._api_key == "test_key"

def test_init_without_api_key():
    """Test initialization without API key raises error."""
    with pytest.raises(ValueError, match="You must provide OpenAI API key"):
        OpenAIEmbeddings(api_key="")

def test_init_with_custom_params():
    """Test initialization with custom parameters."""
    embeddings = OpenAIEmbeddings(
        api_key="test_key",
        model="text-embedding-ada-002",
        max_retries=5
    )
    assert embeddings.model == "text-embedding-ada-002"
    assert embeddings.max_retries == 5

@patch('llmflows.llms.openai_embeddings.call_with_retry')
def test_generate_single_doc(mock_call_with_retry):
    """Test successful embedding generation for a single document."""
    # Mock the API response
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
    mock_call_with_retry.return_value = (mock_response, 0)
    
    embeddings = OpenAIEmbeddings(api_key="test_key")
    doc = VectorDoc(doc="Test document")
    
    result = embeddings.generate(doc)
    
    assert isinstance(result, VectorDoc)
    assert result.embedding == [0.1, 0.2, 0.3]
    mock_call_with_retry.assert_called_once()

@patch('llmflows.llms.openai_embeddings.call_with_retry')
def test_generate_multiple_docs(mock_call_with_retry):
    """Test successful embedding generation for multiple documents."""
    # Mock the API response
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1, 0.2, 0.3]),
        Mock(embedding=[0.4, 0.5, 0.6])
    ]
    mock_call_with_retry.return_value = (mock_response, 0)
    
    embeddings = OpenAIEmbeddings(api_key="test_key")
    docs = [
        VectorDoc(doc="Test document 1"),
        VectorDoc(doc="Test document 2")
    ]
    
    results = embeddings.generate(docs)
    
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].embedding == [0.1, 0.2, 0.3]
    assert results[1].embedding == [0.4, 0.5, 0.6]
    mock_call_with_retry.assert_called_once()

@pytest.mark.asyncio
@patch('llmflows.llms.openai_embeddings.async_call_with_retry')
async def test_generate_async_single_doc(mock_async_call_with_retry):
    """Test async embedding generation for a single document."""
    # Mock the API response
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
    mock_async_call_with_retry.return_value = (mock_response, 0)
    
    embeddings = OpenAIEmbeddings(api_key="test_key")
    doc = VectorDoc(doc="Test document")
    
    result = await embeddings.generate_async(doc)
    
    assert isinstance(result, VectorDoc)
    assert result.embedding == [0.1, 0.2, 0.3]
    mock_async_call_with_retry.assert_called_once()

@pytest.mark.asyncio
@patch('llmflows.llms.openai_embeddings.async_call_with_retry')
async def test_generate_async_multiple_docs(mock_async_call_with_retry):
    """Test async embedding generation for multiple documents."""
    # Mock the API response
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1, 0.2, 0.3]),
        Mock(embedding=[0.4, 0.5, 0.6])
    ]
    mock_async_call_with_retry.return_value = (mock_response, 0)
    
    embeddings = OpenAIEmbeddings(api_key="test_key")
    docs = [
        VectorDoc(doc="Test document 1"),
        VectorDoc(doc="Test document 2")
    ]
    
    results = await embeddings.generate_async(docs)
    
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].embedding == [0.1, 0.2, 0.3]
    assert results[1].embedding == [0.4, 0.5, 0.6]
    mock_async_call_with_retry.assert_called_once()
