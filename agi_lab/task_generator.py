"""
Task Generator
Generates diverse tasks for testing AGI approaches
"""
import random
from typing import Any, Dict, List


class TaskGenerator:
    """Generates tasks for AGI evaluation"""

    @staticmethod
    def generate_world_model_tasks(num_tasks: int = 10) -> List[Dict[str, Any]]:
        """Generate tasks for world model learning"""
        tasks = []

        scenarios = [
            {
                "name": "push_block",
                "states": ["block_left", "block_center", "block_right"],
                "actions": ["push_right", "push_left"],
            },
            {
                "name": "flip_switch",
                "states": ["light_on", "light_off"],
                "actions": ["flip"],
            },
            {
                "name": "open_door",
                "states": ["door_closed", "door_open", "door_locked"],
                "actions": ["open", "close", "lock", "unlock"],
            },
        ]

        for i in range(num_tasks):
            scenario = random.choice(scenarios)

            # Generate trajectory
            trajectory = []
            current_state = random.choice(scenario["states"])

            for _ in range(random.randint(3, 8)):
                action = random.choice(scenario["actions"])

                # Simulate state transitions
                if scenario["name"] == "push_block":
                    if action == "push_right":
                        if current_state == "block_left":
                            next_state = "block_center"
                        elif current_state == "block_center":
                            next_state = "block_right"
                        else:
                            next_state = current_state
                    else:  # push_left
                        if current_state == "block_right":
                            next_state = "block_center"
                        elif current_state == "block_center":
                            next_state = "block_left"
                        else:
                            next_state = current_state

                elif scenario["name"] == "flip_switch":
                    next_state = "light_off" if current_state == "light_on" else "light_on"

                elif scenario["name"] == "open_door":
                    if action == "open" and current_state == "door_closed":
                        next_state = "door_open"
                    elif action == "close" and current_state == "door_open":
                        next_state = "door_closed"
                    elif action == "lock" and current_state == "door_closed":
                        next_state = "door_locked"
                    elif action == "unlock" and current_state == "door_locked":
                        next_state = "door_closed"
                    else:
                        next_state = current_state
                else:
                    next_state = current_state

                trajectory.append({
                    "state": current_state,
                    "action": action,
                })

                current_state = next_state

            trajectory.append({"state": current_state, "action": "done"})

            tasks.append({
                "type": "world_model",
                "scenario": scenario["name"],
                "trajectory": trajectory,
            })

        return tasks

    @staticmethod
    def generate_meta_learning_tasks(num_tasks: int = 10) -> List[Dict[str, Any]]:
        """Generate tasks for meta-learning"""
        tasks = []

        task_types = [
            "classification",
            "regression",
            "pattern_matching",
            "sequence_prediction",
        ]

        for i in range(num_tasks):
            task_type = random.choice(task_types)

            # Generate examples
            examples = []
            for j in range(random.randint(5, 15)):
                if task_type == "classification":
                    # Simple binary classification
                    x = random.random()
                    y = 1 if x > 0.5 else 0
                    examples.append({"input": x, "output": y})

                elif task_type == "regression":
                    x = random.random()
                    y = 2 * x + random.gauss(0, 0.1)  # Linear with noise
                    examples.append({"input": x, "output": y})

                elif task_type == "pattern_matching":
                    pattern = ["A", "B"] * (j % 3 + 1)
                    examples.append({"input": j, "output": pattern})

                elif task_type == "sequence_prediction":
                    seq = list(range(j, j + 5))
                    examples.append({"input": seq[:-1], "output": seq[-1]})

            tasks.append({
                "type": task_type,
                "examples": examples,
            })

        return tasks

    @staticmethod
    def generate_causal_tasks(num_tasks: int = 10) -> List[Dict[str, Any]]:
        """Generate tasks for causal reasoning"""
        tasks = []

        causal_scenarios = [
            {
                "variables": ["rain", "sprinkler", "wet_grass"],
                "edges": [("rain", "wet_grass"), ("sprinkler", "wet_grass")],
            },
            {
                "variables": ["study", "knowledge", "grade"],
                "edges": [("study", "knowledge"), ("knowledge", "grade")],
            },
            {
                "variables": ["exercise", "health", "happiness"],
                "edges": [("exercise", "health"), ("health", "happiness")],
            },
        ]

        for i in range(num_tasks):
            scenario = random.choice(causal_scenarios)

            # Generate observations
            observations = []
            for cause, effect in scenario["edges"]:
                strength = random.uniform(0.5, 1.0)
                observations.append({
                    "cause": cause,
                    "effect": effect,
                    "strength": strength,
                })

            # Generate interventions
            interventions = []
            for cause, effect in scenario["edges"]:
                # Intervention shows true causal effect
                strength = random.uniform(0.6, 1.0)
                interventions.append({
                    "cause": cause,
                    "effect": effect,
                    "strength": strength,
                })

            # Test interventions
            test_interventions = []
            for cause, effect in scenario["edges"]:
                strength = random.uniform(0.5, 1.0)
                test_interventions.append({
                    "cause": cause,
                    "effect": effect,
                    "strength": strength,
                })

            # Counterfactuals
            counterfactuals = []
            for cause, effect in scenario["edges"]:
                counterfactuals.append({
                    "cause": cause,
                    "alt_value": "high",
                    "effect": effect,
                    "expected_effect": "different",
                })

            tasks.append({
                "type": "causal_discovery",
                "observations": observations,
                "interventions": interventions,
                "test_interventions": test_interventions,
                "counterfactuals": counterfactuals,
            })

        return tasks

    @staticmethod
    def generate_transfer_tasks(num_tasks: int = 10) -> List[Dict[str, Any]]:
        """Generate tasks for transfer learning evaluation"""
        # Mix of all task types for transfer
        tasks = []
        tasks.extend(TaskGenerator.generate_world_model_tasks(num_tasks // 3))
        tasks.extend(TaskGenerator.generate_meta_learning_tasks(num_tasks // 3))
        tasks.extend(TaskGenerator.generate_causal_tasks(num_tasks // 3))
        return tasks
