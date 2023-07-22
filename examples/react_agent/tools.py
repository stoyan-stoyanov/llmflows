# pylint: skip-file
import requests


def wikipedia_tool(query: str) -> str:
    """
    Retrieves the introductory paragraph of a Wikipedia article for a given query.

    Args:
        query: A string representing the search term to query on Wikipedia.

    Returns:
        A string representing the introductory paragraph of the article, or a preset
        string if the article was not found.
    """

    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "opensearch",
            "search": query,
            "limit": 1,
            "namespace": 0,
            "format": "json",
        },
    )

    data = response.json()

    if not data[1]:
        result = "Observation: The search didn't return any data"
    else:
        page_title = data[1][0]
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "exintro": True,
                "titles": page_title,
                "format": "json",
                "explaintext": True,
                "exsentences": 5,
            },
        )

        data = response.json()

        # Get page id
        page_id = next(iter(data["query"]["pages"]))

        # Extract the introduction
        result = data["query"]["pages"][page_id].get("extract")
        result = result.replace("\n", "")

    return f"Observation: {result}"


def calculator_tool(calc):
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
    elif "calculator" in thought_action:
        print("Using calculator tool:")
        calc = thought_action.split("calculator:")[1].strip()
        return calculator_tool(calc)

    else:
        return "<invalid_action>"
