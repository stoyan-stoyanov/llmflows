## TL;DR

```python
from llmflows.llms import OpenAI

llm = OpenAI()
result = llm.generate(prompt="Generate a cool title for an 80s rock song")
print(result)

```

***

## Guide

LLMs are one of the main abstractions in LLMFlows. LLM classes are wrappers around LLM APIs such as OpenAI's APIs.
They provide methods for setting up and calling these APIs, retrying failed calls, and formatting the responses.

!!! info

    LLM classes can be imported from `llmfwlos.llms`

```python
from llmflows.llms import OpenAI
```

OpenAI's `GPT-3` is one of the commonly used LLMs, and is available through their completion API. The LLMFlows' `OpenAI` class is a wrapper around this API.
It can be configured in the following way:

```python
llm = OpenAI(
    model="text-davinci-003",
    temperature=0.7,
    max_tokens=500,
    max_retries=3,
)
```

All LLM classes have `.generate()` and `.generate_async()` mehtods that are used for generating text.
In order to generate text with `"text-davinci-003"` the only thing we need to provide is a `prompt`.

```python
result, call_data, model_config = llm.generate(prompt="Generate a cool title for an 80s rock song")
```

The `.generate()` method returns the text completion, the API call information, and the config that was used to make the call;

```python
print(result)
```

```commandline
"Living On The Edge of Time"
```

```python
print(call_data)
```

```commandline
{
   "raw_outputs":"<OpenAIObject text_completion id=cmpl-7YEqPAaoYJCJaFAQ1SGUpOqDsU6tU at 0x11b287310> JSON":{
      "choices":[
         {
            "finish_reason":"stop",
            "index":0,
            "logprobs":null,
            "text":"\n\n\"Living On The Edge of Time\""
         }
      ],
      "created":1688394569,
      "id":"cmpl-7YEqPAaoYJCJaFAQ1SGUpOqDsU6tU",
      "model":"text-davinci-003",
      "object":"text_completion",
      "usage":{
         "completion_tokens":10,
         "prompt_tokens":11,
         "total_tokens":21
      }
   },
   "retries":0
}
```

```python
print(model_config)
```

```commandline
{
   "model_name":"text-davinci-003",
   "temperature":0.7,
   "max_tokens":500
}
```

In the next guide we are going to cover how we can use the `OpenAIChat` class which is an interface for using the chat completion API from OpenAI.
***
[Next: Chat LLMs :material-arrow-right:](Chat LLMs.md){ .md-button }
