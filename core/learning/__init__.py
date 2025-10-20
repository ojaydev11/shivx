"""
ShivX Self-Learning System
==========================
Learns from every interaction to become smarter over time

Includes:
- Experience Replay (existing)
- Continuous Web Learning (existing)
- Transfer Learning (Phase 3)
- Curriculum Learning (Phase 3)
"""

from .experience_replay import ExperienceReplay, get_experience_replay

try:
    from .continuous_web_learner import ContinuousWebLearner, get_continuous_web_learner
    WEB_LEARNER_AVAILABLE = True
except ImportError:
    WEB_LEARNER_AVAILABLE = False

# Phase 3 additions
try:
    from .transfer_learner import (
        TransferLearner,
        Task,
        TransferResult,
        MAMLModel,
        PrototypicalNetwork,
        quick_transfer,
    )
    from .curriculum import (
        CurriculumManager,
        CurriculumBuilder,
        CurriculumScheduler,
        Skill,
        CurriculumStage,
        LearningExample,
        DifficultyLevel,
        SchedulingStrategy,
        quick_curriculum_test,
    )
    PHASE3_AVAILABLE = True
except ImportError:
    PHASE3_AVAILABLE = False

# Phase 4 additions
try:
    from .continual_learner import (
        ContinualLearner,
        EWCRegularizer,
        MemoryBuffer,
        quick_continual_test,
    )
    from .active_learner import (
        ActiveLearner,
        QueryStrategy,
        UncertaintySampler,
        QueryByCommittee,
        DiversitySampler,
        quick_active_learning_test,
    )
    from .self_supervised import (
        SelfSupervisedLearner,
        PretrainingTask,
        ContrastiveLearner,
        MaskedPredictor,
        Autoencoder,
        quick_self_supervised_test,
    )
    PHASE4_AVAILABLE = True
except ImportError:
    PHASE4_AVAILABLE = False

# Build __all__ dynamically
__all__ = [
    'ExperienceReplay',
    'get_experience_replay',
]

if WEB_LEARNER_AVAILABLE:
    __all__.extend([
        'ContinuousWebLearner',
        'get_continuous_web_learner',
    ])

if PHASE3_AVAILABLE:
    __all__.extend([
        'TransferLearner',
        'Task',
        'TransferResult',
        'MAMLModel',
        'PrototypicalNetwork',
        'quick_transfer',
        'CurriculumManager',
        'CurriculumBuilder',
        'CurriculumScheduler',
        'Skill',
        'CurriculumStage',
        'LearningExample',
        'DifficultyLevel',
        'SchedulingStrategy',
        'quick_curriculum_test',
    ])

if PHASE4_AVAILABLE:
    __all__.extend([
        'ContinualLearner',
        'EWCRegularizer',
        'MemoryBuffer',
        'quick_continual_test',
        'ActiveLearner',
        'QueryStrategy',
        'UncertaintySampler',
        'QueryByCommittee',
        'DiversitySampler',
        'quick_active_learning_test',
        'SelfSupervisedLearner',
        'PretrainingTask',
        'ContrastiveLearner',
        'MaskedPredictor',
        'Autoencoder',
        'quick_self_supervised_test',
    ])
