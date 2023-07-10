import unittest
from typing import Any
from llmflows.flows.base_flowstep import BaseFlowStep


class TestBaseFlowStep(unittest.TestCase):
    class DummyFlowStep(BaseFlowStep):
        def generate(self, inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
            return "dummy_result", "dummy_call_data", "dummy_config"

    def setUp(self):
        self.base_flow_step = self.DummyFlowStep("TestStep", "TestKey", None)

    def test_init(self):
        self.assertEqual(self.base_flow_step.name, "TestStep")
        self.assertEqual(self.base_flow_step.output_key, "TestKey")
        self.assertEqual(self.base_flow_step.callbacks, [])

    def test_connect(self):
        step1 = self.DummyFlowStep("Step1", "Key1", None)
        step2 = self.DummyFlowStep("Step2", "Key2", None)
        self.base_flow_step.connect(step1, step2)

        self.assertEqual(self.base_flow_step.next_steps, [step1, step2])
        self.assertEqual(step1.parents, [self.base_flow_step])
        self.assertEqual(step2.parents, [self.base_flow_step])

    def test_connect_same_output_key(self):
        step1 = self.DummyFlowStep("Step1", "Key1", None)
        step2 = self.DummyFlowStep("Step2", "Key1", None)
        with self.assertRaises(ValueError):
            self.base_flow_step.connect(step1, step2)

    def test_connect_cycle(self):
        step1 = self.DummyFlowStep("Step1", "Key1", None)
        step2 = self.DummyFlowStep("Step2", "Key2", None)
        step1.connect(step2)
        step2.connect(self.base_flow_step)
        with self.assertRaises(ValueError):
            self.base_flow_step.connect(step1)
