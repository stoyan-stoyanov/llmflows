# pylint: skip-file

import unittest
from llmflows.flows.functional_flowstep import FunctionalFlowStep
from llmflows.callbacks.base_callback import BaseCallback


def flowstep_function(arg1, arg2):
    return "Test output"


class TestFunctionalFlowStep(unittest.TestCase):
    def setUp(self):
        self.functional_flowstep = FunctionalFlowStep(
            "TestStep", flowstep_function, "output_key"
        )

    def test_init(self):
        self.assertEqual(self.functional_flowstep.name, "TestStep")
        self.assertEqual(self.functional_flowstep.output_key, "output_key")
        self.assertEqual(self.functional_flowstep.callbacks, [])
        self.assertEqual(self.functional_flowstep.required_keys, ["arg1", "arg2"])
        self.assertEqual(self.functional_flowstep.flowstep_fn, flowstep_function)

    def test_generate(self):
        inputs = {"arg1": "Test input", "arg2": "Another test input"}
        output, call_data, config = self.functional_flowstep.generate(inputs)
        self.assertEqual(output, "Test output")
        self.assertEqual(call_data, None)
        self.assertEqual(config, None)

    def test_generate_with_required_inputs(self):
        inputs = {"arg1": "Test input", "arg2": "Another test input"}
        output, call_data, config = self.functional_flowstep.generate(inputs)
        self.assertEqual(output, "Test output")
        self.assertEqual(call_data, None)
        self.assertEqual(config, None)

    def test_generate_with_additional_unneeded_keys(self):
        inputs = {
            "arg1": "Test input",
            "arg2": "Another test input",
            "arg3": "Unneeded test input",
        }
        output, call_data, config = self.functional_flowstep.generate(inputs)
        self.assertEqual(output, "Test output")
        self.assertEqual(call_data, None)
        self.assertEqual(config, None)

    def test_generate_with_missing_required_keys(self):
        inputs = {"arg1": "Test input"}
        with self.assertRaises(Exception) as context:
            self.functional_flowstep.generate(inputs)
        self.assertTrue(
            "missing 1 required positional argument" in str(context.exception)
        )

    def test_generate_with_non_string_output(self):
        def non_string_output_function(arg1, arg2):
            return 12345

        non_string_output_flowstep = FunctionalFlowStep(
            "TestStep", non_string_output_function, "output_key"
        )
        inputs = {"arg1": "Test input", "arg2": "Another test input"}
        with self.assertRaises(TypeError) as context:
            non_string_output_flowstep.generate(inputs)
        self.assertEqual(
            str(context.exception), "Return value must be of type str, but got int"
        )
