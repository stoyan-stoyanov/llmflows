# pylint: skip-file
from mediawiki import MediaWiki

wikipedia = MediaWiki()

def wikipedia_tool(query: str) -> str:
    """
    Retrieves the summary of a Wikipedia article for a given query.

    Args:
        query: A string representing the title of a Wikipedia article.

    Returns:
        A string representing the summary of the article, or a preset string if the 
        article was not found.
    """
    try:
        wikipedia_page = wikipedia.page(query)
        return f"Observation: {wikipedia_page.summary}"
    except:
        return "Observation: The search didn't return any data"


def calculator_tool(calc) -> str:
    """
    A simple calculator tool that uses eval() to calculate the result of a given
    expression.
    """
    return "Observation: the calculation result is " + str(eval(calc))


def tool_selector(thought_action: str) -> str:
    """Invokes a tool based on the action specified in the agent output"""
    if "final answer:" in thought_action:
        return "<final_answer>"
    elif "wikipedia:" in thought_action:
        print("Using wikipedia tool:")
        question = thought_action.split("wikipedia:")[1].strip()
        return wikipedia_tool(question)
    elif "calculator:" in thought_action:
        print("Using calculator tool:")
        calc = thought_action.split("calculator:")[1].strip()
        return calculator_tool(calc)

    else:
        return "<invalid_action>"
