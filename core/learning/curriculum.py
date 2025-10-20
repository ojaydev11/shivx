"""
Curriculum Learning - Progressive skill building through structured learning.

Just like humans learn to walk before they run, ShivX learns:
- Simple tasks before complex ones
- Prerequisites before advanced skills
- Easy examples before hard ones

This enables faster, more stable learning compared to random training.

Part of ShivX 6/10 AGI transformation (Phase 3).
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import heapq

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Difficulty levels for curriculum stages"""
    TRIVIAL = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


class SchedulingStrategy(Enum):
    """Strategies for curriculum scheduling"""
    LINEAR = "linear"  # Progress linearly through stages
    PERFORMANCE_BASED = "performance_based"  # Advance when performance threshold met
    ADAPTIVE = "adaptive"  # Dynamically adjust based on learning rate
    SELF_PACED = "self_paced"  # Let model choose difficulty


@dataclass
class Skill:
    """Represents a learnable skill"""
    id: str
    name: str
    description: str
    difficulty: DifficultyLevel
    prerequisites: List[str] = field(default_factory=list)  # Required skills
    learning_resources: List[Dict[str, Any]] = field(default_factory=list)
    mastery_threshold: float = 0.8  # Performance threshold for mastery
    current_performance: float = 0.0
    practice_count: int = 0


@dataclass
class CurriculumStage:
    """A stage in the curriculum"""
    id: str
    name: str
    skills: List[Skill]
    difficulty: DifficultyLevel
    estimated_duration_hours: float
    completion_criteria: Dict[str, Any]  # What defines completion

    # Progress tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempts: int = 0
    current_performance: float = 0.0


@dataclass
class LearningExample:
    """A training example with difficulty annotation"""
    id: str
    content: Dict[str, Any]
    difficulty: float  # 0-1, where 1 is hardest
    concepts: List[str]  # Concepts covered in this example
    success_rate: float = 0.0  # Historical success rate
    presentation_count: int = 0


class CurriculumScheduler:
    """
    Schedules curriculum progression.

    Decides when to move to next stage based on performance.
    """

    def __init__(
        self,
        strategy: SchedulingStrategy = SchedulingStrategy.PERFORMANCE_BASED,
        performance_threshold: float = 0.75,
        min_attempts_per_stage: int = 10,
    ):
        """
        Initialize curriculum scheduler.

        Args:
            strategy: Scheduling strategy
            performance_threshold: Performance required to advance
            min_attempts_per_stage: Minimum attempts before advancing
        """
        self.strategy = strategy
        self.performance_threshold = performance_threshold
        self.min_attempts_per_stage = min_attempts_per_stage

        logger.info(f"Curriculum Scheduler: {strategy.value}, threshold={performance_threshold}")

    def should_advance(
        self,
        stage: CurriculumStage,
        recent_performance: List[float],
    ) -> bool:
        """
        Decide if should advance to next stage.

        Args:
            stage: Current stage
            recent_performance: Recent performance scores

        Returns:
            True if should advance
        """
        if stage.attempts < self.min_attempts_per_stage:
            return False

        if not recent_performance:
            return False

        avg_performance = np.mean(recent_performance[-10:])  # Last 10 attempts

        if self.strategy == SchedulingStrategy.LINEAR:
            # Just check minimum attempts
            return True

        elif self.strategy == SchedulingStrategy.PERFORMANCE_BASED:
            # Check if performance meets threshold
            return avg_performance >= self.performance_threshold

        elif self.strategy == SchedulingStrategy.ADAPTIVE:
            # Check learning rate (improvement over time)
            if len(recent_performance) < 5:
                return False

            # Compare recent vs earlier performance
            early_perf = np.mean(recent_performance[:5])
            recent_perf = np.mean(recent_performance[-5:])

            improvement = recent_perf - early_perf

            # Advance if plateaued (improvement < 0.05) or reached threshold
            return improvement < 0.05 or recent_perf >= self.performance_threshold

        elif self.strategy == SchedulingStrategy.SELF_PACED:
            # Let model choose (for now, use performance-based)
            return avg_performance >= self.performance_threshold

        return False


class DifficultyEstimator:
    """
    Estimates difficulty of examples.

    Uses multiple signals to determine difficulty.
    """

    def __init__(self):
        self.example_history: Dict[str, List[bool]] = {}  # example_id -> [successes]

    def estimate_difficulty(
        self,
        example: LearningExample,
        features: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Estimate difficulty of an example.

        Args:
            example: Learning example
            features: Additional features for estimation

        Returns:
            Difficulty score (0-1)
        """
        # Start with annotated difficulty if available
        if hasattr(example, 'difficulty') and example.difficulty is not None:
            base_difficulty = example.difficulty
        else:
            base_difficulty = 0.5

        # Adjust based on historical success rate
        if example.success_rate > 0:
            # Low success rate → high difficulty
            difficulty_from_success = 1.0 - example.success_rate
            base_difficulty = 0.5 * base_difficulty + 0.5 * difficulty_from_success

        # Adjust based on complexity (if features provided)
        if features:
            complexity_signals = []

            # Length/size
            if "length" in features:
                # Longer examples are typically harder
                complexity_signals.append(min(features["length"] / 1000.0, 1.0))

            # Number of concepts
            if "num_concepts" in features:
                complexity_signals.append(min(features["num_concepts"] / 10.0, 1.0))

            # Nesting depth
            if "depth" in features:
                complexity_signals.append(min(features["depth"] / 5.0, 1.0))

            if complexity_signals:
                complexity_score = np.mean(complexity_signals)
                base_difficulty = 0.7 * base_difficulty + 0.3 * complexity_score

        return float(np.clip(base_difficulty, 0.0, 1.0))

    def update_difficulty(
        self,
        example: LearningExample,
        success: bool,
    ):
        """Update difficulty estimate based on outcome"""
        if example.id not in self.example_history:
            self.example_history[example.id] = []

        self.example_history[example.id].append(success)

        # Update success rate
        example.presentation_count += 1
        example.success_rate = np.mean(self.example_history[example.id])


class CurriculumBuilder:
    """
    Builds curriculum from skills and examples.

    Organizes learning into progressive stages.
    """

    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.examples: Dict[str, LearningExample] = {}
        self.difficulty_estimator = DifficultyEstimator()

    def add_skill(self, skill: Skill):
        """Add a skill to the curriculum"""
        self.skills[skill.id] = skill
        logger.info(f"Added skill: {skill.name} (difficulty: {skill.difficulty.name})")

    def add_example(self, example: LearningExample):
        """Add a learning example"""
        self.examples[example.id] = example

    def build_curriculum(
        self,
        goal_skills: List[str],
    ) -> List[CurriculumStage]:
        """
        Build curriculum to learn goal skills.

        Uses topological sort on prerequisite graph.

        Args:
            goal_skills: List of skill IDs to learn

        Returns:
            Ordered list of curriculum stages
        """
        # Build dependency graph
        required_skills = self._get_required_skills(goal_skills)

        # Topological sort to determine order
        skill_order = self._topological_sort(required_skills)

        # Group into stages by difficulty
        stages = self._group_into_stages(skill_order)

        logger.info(f"Built curriculum: {len(stages)} stages, {len(required_skills)} skills")

        return stages

    def _get_required_skills(self, goal_skills: List[str]) -> Set[str]:
        """Get all required skills (including prerequisites)"""
        required = set()
        to_process = goal_skills.copy()

        while to_process:
            skill_id = to_process.pop()

            if skill_id in required:
                continue

            if skill_id not in self.skills:
                logger.warning(f"Unknown skill: {skill_id}")
                continue

            required.add(skill_id)

            skill = self.skills[skill_id]
            to_process.extend(skill.prerequisites)

        return required

    def _topological_sort(self, skill_ids: Set[str]) -> List[str]:
        """Topological sort of skills by prerequisites"""
        # Build adjacency list
        graph = {skill_id: [] for skill_id in skill_ids}
        in_degree = {skill_id: 0 for skill_id in skill_ids}

        for skill_id in skill_ids:
            skill = self.skills[skill_id]
            for prereq in skill.prerequisites:
                if prereq in skill_ids:
                    graph[prereq].append(skill_id)
                    in_degree[skill_id] += 1

        # Kahn's algorithm
        queue = [skill_id for skill_id in skill_ids if in_degree[skill_id] == 0]
        result = []

        while queue:
            # Sort by difficulty (easier first)
            queue.sort(key=lambda s: self.skills[s].difficulty.value)

            skill_id = queue.pop(0)
            result.append(skill_id)

            for neighbor in graph[skill_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(skill_ids):
            logger.warning("Circular dependencies detected in skill graph")

        return result

    def _group_into_stages(self, skill_order: List[str]) -> List[CurriculumStage]:
        """Group skills into curriculum stages"""
        stages = []
        current_stage_skills = []
        current_difficulty = None

        for skill_id in skill_order:
            skill = self.skills[skill_id]

            # Start new stage if difficulty changes
            if current_difficulty is not None and skill.difficulty != current_difficulty:
                if current_stage_skills:
                    stage = CurriculumStage(
                        id=f"stage_{len(stages)}",
                        name=f"Stage {len(stages) + 1}: {current_difficulty.name}",
                        skills=current_stage_skills.copy(),
                        difficulty=current_difficulty,
                        estimated_duration_hours=len(current_stage_skills) * 2.0,
                        completion_criteria={"all_skills_mastered": True},
                    )
                    stages.append(stage)
                    current_stage_skills = []

            current_difficulty = skill.difficulty
            current_stage_skills.append(skill)

        # Add final stage
        if current_stage_skills:
            stage = CurriculumStage(
                id=f"stage_{len(stages)}",
                name=f"Stage {len(stages) + 1}: {current_difficulty.name}",
                skills=current_stage_skills,
                difficulty=current_difficulty,
                estimated_duration_hours=len(current_stage_skills) * 2.0,
                completion_criteria={"all_skills_mastered": True},
            )
            stages.append(stage)

        return stages


class CurriculumManager:
    """
    Main curriculum learning system.

    Manages progression through curriculum stages.
    """

    def __init__(
        self,
        builder: Optional[CurriculumBuilder] = None,
        scheduler: Optional[CurriculumScheduler] = None,
    ):
        """
        Initialize Curriculum Manager.

        Args:
            builder: Curriculum builder
            scheduler: Curriculum scheduler
        """
        self.builder = builder or CurriculumBuilder()
        self.scheduler = scheduler or CurriculumScheduler()

        self.curriculum: List[CurriculumStage] = []
        self.current_stage_idx: int = 0
        self.performance_history: List[float] = []

        logger.info("Curriculum Manager initialized")

    def set_curriculum(self, goal_skills: List[str]):
        """Set curriculum based on goal skills"""
        self.curriculum = self.builder.build_curriculum(goal_skills)
        self.current_stage_idx = 0

        logger.info(f"Curriculum set: {len(self.curriculum)} stages")

    def get_current_stage(self) -> Optional[CurriculumStage]:
        """Get current curriculum stage"""
        if self.current_stage_idx < len(self.curriculum):
            return self.curriculum[self.current_stage_idx]
        return None

    def get_next_examples(
        self,
        batch_size: int = 32,
        difficulty_range: Optional[Tuple[float, float]] = None,
    ) -> List[LearningExample]:
        """
        Get next batch of examples to learn from.

        Args:
            batch_size: Number of examples
            difficulty_range: (min, max) difficulty range

        Returns:
            List of learning examples
        """
        stage = self.get_current_stage()

        if stage is None:
            logger.info("Curriculum complete")
            return []

        # Get examples for current stage skills
        relevant_examples = []

        for skill in stage.skills:
            for resource in skill.learning_resources:
                if "example_id" in resource:
                    example_id = resource["example_id"]
                    if example_id in self.builder.examples:
                        example = self.builder.examples[example_id]

                        # Filter by difficulty if specified
                        if difficulty_range:
                            min_diff, max_diff = difficulty_range
                            if not (min_diff <= example.difficulty <= max_diff):
                                continue

                        relevant_examples.append(example)

        # Sample batch
        if len(relevant_examples) <= batch_size:
            return relevant_examples
        else:
            # Prioritize examples with lower success rate (need more practice)
            weights = [1.0 - ex.success_rate + 0.1 for ex in relevant_examples]
            weights = np.array(weights) / sum(weights)

            indices = np.random.choice(
                len(relevant_examples),
                size=batch_size,
                replace=False,
                p=weights,
            )

            return [relevant_examples[i] for i in indices]

    def record_performance(self, performance: float):
        """Record performance on current stage"""
        self.performance_history.append(performance)

        stage = self.get_current_stage()
        if stage:
            stage.attempts += 1
            stage.current_performance = performance

            # Update skill performance
            for skill in stage.skills:
                skill.practice_count += 1
                skill.current_performance = 0.7 * skill.current_performance + 0.3 * performance

    def should_advance_stage(self) -> bool:
        """Check if should advance to next stage"""
        stage = self.get_current_stage()

        if stage is None:
            return False

        return self.scheduler.should_advance(stage, self.performance_history)

    def advance_stage(self):
        """Advance to next curriculum stage"""
        stage = self.get_current_stage()

        if stage:
            stage.completed_at = datetime.utcnow()
            logger.info(
                f"Completed stage: {stage.name} "
                f"(performance: {stage.current_performance:.3f}, attempts: {stage.attempts})"
            )

        self.current_stage_idx += 1
        self.performance_history = []  # Reset for new stage

        next_stage = self.get_current_stage()
        if next_stage:
            next_stage.started_at = datetime.utcnow()
            logger.info(f"Started stage: {next_stage.name}")

    def get_progress(self) -> Dict[str, Any]:
        """Get curriculum progress"""
        total_stages = len(self.curriculum)
        completed_stages = self.current_stage_idx

        current_stage = self.get_current_stage()

        return {
            "total_stages": total_stages,
            "completed_stages": completed_stages,
            "current_stage": current_stage.name if current_stage else "Complete",
            "progress_percent": (completed_stages / total_stages * 100) if total_stages > 0 else 100,
            "current_performance": current_stage.current_performance if current_stage else 1.0,
            "skills_mastered": sum(
                1 for skill in self.builder.skills.values()
                if skill.current_performance >= skill.mastery_threshold
            ),
            "total_skills": len(self.builder.skills),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics"""
        progress = self.get_progress()

        avg_performance = np.mean(self.performance_history) if self.performance_history else 0.0

        return {
            **progress,
            "avg_recent_performance": avg_performance,
            "total_attempts": sum(stage.attempts for stage in self.curriculum),
        }


# Convenience function
def quick_curriculum_test(
    skills: List[Skill],
    goal_skill_ids: List[str],
) -> List[CurriculumStage]:
    """
    Quick curriculum building test.

    Args:
        skills: List of skills
        goal_skill_ids: Goal skills to learn

    Returns:
        Curriculum stages
    """
    builder = CurriculumBuilder()

    for skill in skills:
        builder.add_skill(skill)

    curriculum = builder.build_curriculum(goal_skill_ids)

    return curriculum
