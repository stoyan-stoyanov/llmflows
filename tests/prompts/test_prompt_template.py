# pylint: skip-file

import unittest
from llmflows.prompts.prompt_template import PromptTemplate


class TestPromptTemplate(unittest.TestCase):
    def test_init(self):
        # Initialize a PromptTemplate instance with a string containing no variables
        template = PromptTemplate("Hello, world!")
        self.assertEqual(template.prompt, "Hello, world!")
        self.assertEqual(template.variables, set())

        # Initialize a PromptTemplate instance with a string containing variables
        template = PromptTemplate("Hello, {name}!")
        self.assertEqual(template.prompt, "Hello, {name}!")
        self.assertEqual(template.variables, {"name"})

    def test_get_prompt_without_variables(self):
        template = PromptTemplate("Hello, world!")
        self.assertEqual(template.get_prompt(), "Hello, world!")

    def test_get_prompt_with_variables(self):
        template = PromptTemplate("Hello, {name}!")
        self.assertEqual(template.get_prompt(name="John"), "Hello, John!")

    def test_get_prompt_with_multiple_variables(self):
        template = PromptTemplate("Hello, {first_name} {last_name}!")
        self.assertEqual(
            template.get_prompt(first_name="John", last_name="Doe"), "Hello, John Doe!"
        )

    def test_get_prompt_with_missing_variables(self):
        template = PromptTemplate("Hello, {name}!")
        with self.assertRaises(ValueError):
            template.get_prompt()

    def test_get_prompt_with_extra_variables(self):
        template = PromptTemplate("Hello, {name}!")
        with self.assertRaises(ValueError):
            template.get_prompt(name="John", age="30")

    def test_unmatched_variables(self):
        template = PromptTemplate("Hello, {name}!")
        with self.assertRaises(ValueError):
            template.get_prompt(age="25")

    def test_empty_string(self):
        template = PromptTemplate("")
        self.assertEqual(template.get_prompt(), "")

    def test_special_characters(self):
        template = PromptTemplate("Hello, {name}! #$%&*")
        self.assertEqual(template.get_prompt(name="John"), "Hello, John! #$%&*")

    def test_escape_curly_braces(self):
        template = PromptTemplate("Hello, {{name}}!")
        self.assertEqual(template.get_prompt(), "Hello, {{name}}!")

    def test_variable_replacement_multiple_times(self):
        template = PromptTemplate("Hello, {name}! Nice to meet you, {name}!")
        self.assertEqual(
            template.get_prompt(name="John"), "Hello, John! Nice to meet you, John!"
        )

    def test_string_with_only_variable(self):
        template = PromptTemplate("{greeting}")
        self.assertEqual(template.get_prompt(greeting="Hello, World!"), "Hello, World!")

    def test_variables_with_number(self):
        template = PromptTemplate("Hello, {name1} and {name2}!")
        self.assertEqual(
            template.get_prompt(name1="John", name2="Jane"), "Hello, John and Jane!"
        )
