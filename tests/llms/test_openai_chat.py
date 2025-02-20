"""Tests for the OpenAI chat implementation."""

import pytest
from unittest.mock import Mock, patch
from openai import APIError, Timeout, RateLimitError, APIConnectionError
from llmflows.llms.openai_chat import OpenAIChat
from llmflows.llms.message_history import MessageHistory

def test_init_with_valid_api_key():
    """Test initialization with valid API key."""
    chat = OpenAIChat(api_key="test_key")
    assert chat.model == "gpt-4o-mini"
    assert chat.temperature == 0.7
    assert chat.max_retries == 3
    assert chat.verbose is False
    assert chat.tools is None
    assert chat.tools_functions is None
    assert chat.response_format is None
    assert chat.prediction is None

def test_init_without_api_key():
    """Test initialization without API key raises error."""
    with pytest.raises(ValueError, match="You must provide OpenAI API key"):
        OpenAIChat(api_key="")

def test_init_with_custom_params():
    """Test initialization with custom parameters."""
    chat = OpenAIChat(
        api_key="test_key",
        model="gpt-4",
        temperature=0.5,
        max_retries=5,
        verbose=True
    )
    assert chat.model == "gpt-4"
    assert chat.temperature == 0.5
    assert chat.max_retries == 5
    assert chat.verbose is True

def test_multiple_modes_error():
    """Test that using multiple modes raises error."""
    with pytest.raises(
        ValueError, 
        match="You can not use more than one of the following modes"
    ):
        OpenAIChat(
            api_key="test_key",
            tools=[{"function": {"name": "test"}}],
            response_format={"type": "json"},
        )

def test_tools_without_functions_error():
    """Test that using tools without functions raises error."""
    with pytest.raises(ValueError, match="If you want to use function .tools. calling"):
        OpenAIChat(
            api_key="test_key",
            tools=[{"function": {"name": "test"}}]  # Don't pass tools_functions at all
        )

def example_tool_function(param1: str, param2: int):
    """Example tool function for testing."""
    return f"{param1}: {param2}"

def test_tools_validation_success():
    """Test successful tools validation."""
    tools = [{
        "function": {
            "name": "example_tool_function",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                    "param2": {"type": "integer"}
                },
                "required": ["param1", "param2"]
            }
        }
    }]

    chat = OpenAIChat(
        api_key="test_key",
        tools=tools,
        tools_functions=[example_tool_function]
    )
    assert chat.tools == tools
    assert "example_tool_function" in chat.tools_functions

@patch('llmflows.llms.openai_chat.call_with_retry')
def test_generate_success(mock_call_with_retry):
    """Test successful message generation."""
    # Mock the API response
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_call_with_retry.return_value = (mock_response, 0)
    
    chat = OpenAIChat(api_key="test_key")
    message_history = MessageHistory()
    message_history.add_user_message("Test message")
    
    response, call_data, model_config = chat.generate(message_history)
    
    assert response == "Test response"
    assert call_data["retries"] == 0
    assert model_config["model_name"] == "gpt-4o-mini"
    assert model_config["temperature"] == 0.7

@patch('llmflows.llms.openai_chat.call_with_retry')
def test_generate_with_retries(mock_call_with_retry):
    """Test message generation with retries."""
    # Mock the API response with retries
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_call_with_retry.return_value = (mock_response, 2)
    
    chat = OpenAIChat(api_key="test_key")
    message_history = MessageHistory()
    message_history.add_user_message("Test message")
    
    response, call_data, model_config = chat.generate(message_history)
    
    assert response == "Test response"
    assert call_data["retries"] == 2

@pytest.mark.asyncio
@patch('llmflows.llms.openai_chat.async_call_with_retry')
async def test_generate_async(mock_async_call_with_retry):
    """Test async message generation."""
    # Mock the API response
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_async_call_with_retry.return_value = (mock_response, 0)
    
    chat = OpenAIChat(api_key="test_key")
    message_history = MessageHistory()
    message_history.add_user_message("Test message")
    
    response, call_data, model_config = await chat.generate_async(message_history)
    
    assert response == "Test response"
    assert call_data["retries"] == 0
    assert model_config["model_name"] == "gpt-4o-mini"
