# pylint: disable=R0913, R0902, R0801, W0221
"""
This module implements a wrapper for OpenAI chat models, using BaseLLM as a 
base class.
"""


class MessageHistory:
    """
    """

    def __init__(
        self,
        max_messages: int = 0,
    ):
        self.max_messages = max_messages
        self.messages = []

    @property
    def system_prompt(self) -> str:
        """
        Returns the system prompt.

        Returns:
            str: The system prompt content.
        """
        if self.messages and self.messages[0]["role"] == "system":
            return self.messages[0]["content"]
        return ""  # return an empty string if there is no system prompt

    @system_prompt.setter
    def system_prompt(self, new_prompt: str) -> None:
        """
        Sets the system prompt.

        Args:
            new_prompt (str): The new system prompt.
        """
        if not self.messages or self.messages[0]["role"] != "system":
            self.messages.insert(0, {"role": "system", "content": new_prompt})
        else:
            self.update_system_prompt(new_prompt)


    def update_system_prompt(self, new_prompt: str):
        """
        Updates the system prompt sent to the chat API.

        Args:
            new_prompt (str): The new system prompt.
        """
        self.messages[0] = {"role": "system", "content": new_prompt}

    @property
    def messages(self):
        """
        Returns the conversation history.
        """
        return self._messages

    @messages.setter
    def messages(self, value):
        """
        Sets the conversation history.

        Each message in the list should be a dictionary containing 
        "role" and "content" keys.

        Args:
            value (list): A list of message dictionaries.

        Raises:
            ValueError: If the provided value is not a list or if any 
            dictionary in the list is not a valid message.
        """
        if not isinstance(value, list):
            raise ValueError("messages must be a list of dicts")
        for item in value:
            self.validate_message(item)
        self._messages = value

    def add_user_message(self, message: str) -> None:
        self.add_message(message_str=message, role="user")
    
    def add_ai_message(self, message: str) -> None:
        self.add_message(message_str=message, role="assistant")

    def add_message(self, message_str: str, role: str) -> None:
        """
        Appends a new message to the message history.

        Args:
            message_str (str): Content of the message.
            role (str, optional): Role in the conversation. Can be "user" or
                "assistant". Defaults to "user".

        Returns:
            Updated conversation history.
        """
        role = self.validate_role(role)

        if self.max_messages and (len(self.messages) >= self.max_messages):
            self.remove_message(idx=1)

        self.messages.append({"role": role, "content": message_str})


    @staticmethod
    def validate_role(role: str) -> str:
        """
        Validates the role of a message.

        Args:
            role (str): The role of the message (either "user", "assistant", 
                or "system").

        Returns:
            str: The validated role.

        Raises:
            ValueError: If the role is "system", as this should be updated using 
                'update_system_prompt' method.
            ValueError: If the role is not "user" or "assistant".
        """
        # if role == "system":
        #     raise ValueError(
        #       "To update the system prompt use the 'update_system_prompt method"
        # )

        if role not in ["user", "system", "assistant"]:
            raise ValueError(
                "The role should be either 'system', 'user' or 'assistant'"
            )

        return role

    def validate_message(self, message: dict[str, str]) -> dict[str, str]:
        """
        Validates a message for required fields.

        Args:
            message (dict[str, str]): The message to validate.

        Returns:
            The validated message.

        Raises:
            ValueError: If the provided message does not contain the necessary fields 
                ("role" and "content").
        """
        if {"role", "content"}.issubset(message):
            self.validate_role(message["role"])
            return message
        raise ValueError("The provided message is not a valid message.")

    def replace_message(self, new_message: str, new_role: str, idx: int):
        """
        Replaces a message in the list of messages sent to the chat API.

        Args:
            new_message (str): The new message to replace the old message with.
            idx (int): The index of the message to replace.
        """
        message = {"role": new_role, "content": new_message}
        self.validate_message(message)
        self.messages[idx] = message

    def remove_message(self, idx=-1):
        """
        Removes a message from the list of messages sent to the chat API.

        Args:
            idx (int): The index of the message to remove.
        """
        self.messages.pop(idx)
