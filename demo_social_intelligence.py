#!/usr/bin/env python3
"""
Demonstration of Pillar 9: Social Intelligence & Theory of Mind

This script demonstrates the comprehensive capabilities of the social intelligence system:
- Theory of Mind: Modeling other agents' mental states
- Social Reasoning: Understanding norms and predicting behavior
- Collaboration: Cooperative task execution and conflict resolution
"""

from agi_engine.social import (
    TheoryOfMind,
    SocialReasoner,
    CollaborationEngine,
    Belief,
    BeliefType,
    Intention,
    IntentionType,
    NormType,
    SocialRole,
    BehaviorType,
    CommunicationType,
    ConflictType,
    TaskStatus,
)

def demo_theory_of_mind():
    """Demonstrate Theory of Mind capabilities"""
    print("=" * 80)
    print("THEORY OF MIND DEMONSTRATION")
    print("=" * 80)

    tom = TheoryOfMind()

    # Register other agents
    print("\n1. Registering agents and building mental models...")
    alice = tom.register_agent(
        agent_id="alice_001",
        name="Alice",
        capabilities={"vision", "hearing", "communication"}
    )

    bob = tom.register_agent(
        agent_id="bob_001",
        name="Bob",
        capabilities={"vision", "manipulation", "planning"}
    )

    print(f"   ✓ Registered {alice.name} with capabilities: {alice.capabilities}")
    print(f"   ✓ Registered {bob.name} with capabilities: {bob.capabilities}")

    # Observe behaviors and infer intentions
    print("\n2. Observing behaviors and inferring intentions...")

    intention1 = tom.observe_behavior(
        agent_id="alice_001",
        behavior="Move toward the door",
        context={"target": "door", "speed": "normal"}
    )

    intention2 = tom.observe_behavior(
        agent_id="bob_001",
        behavior="Say 'I need help with this task'",
        context={"task": "lifting heavy object"}
    )

    print(f"   ✓ Alice's inferred intention: {intention1.description if intention1 else 'None'}")
    print(f"   ✓ Bob's inferred intention: {intention2.description if intention2 else 'None'}")

    # Predict behavior
    print("\n3. Predicting future behaviors...")
    predictions_alice = tom.predict_behavior("alice_001", {"location": "near door"})
    print(f"   ✓ Alice's predicted behaviors:")
    for behavior, probability in predictions_alice[:3]:
        print(f"     - {behavior} (probability: {probability:.2f})")

    # Take perspective
    print("\n4. Taking another agent's perspective...")
    perspective = tom.take_perspective("bob_001", "Heavy object needs to be moved")
    print(f"   ✓ Bob's perspective:")
    print(f"     - Perception: {perspective['perception']}")
    print(f"     - Emotion: {perspective['emotion'].value}")
    print(f"     - Intentions: {perspective['intentions']}")

    # Communication updates mental model
    print("\n5. Updating mental models from communication...")
    tom.update_from_communication(
        agent_id="alice_001",
        message="I believe the door is locked",
        context={"topic": "door status"}
    )

    summary = tom.get_agent_summary("alice_001")
    print(f"   ✓ Alice's mental model updated:")
    print(f"     - Beliefs: {summary['beliefs_count']}")
    print(f"     - Current emotion: {summary['current_emotion']}")
    print(f"     - Trust level: {summary['trust_level']:.2f}")

    print("\n   ✅ Theory of Mind demonstration complete!")
    return tom


def demo_social_reasoning():
    """Demonstrate Social Reasoning capabilities"""
    print("\n" + "=" * 80)
    print("SOCIAL REASONING DEMONSTRATION")
    print("=" * 80)

    reasoner = SocialReasoner()

    # Show pre-loaded norms
    print("\n1. Built-in social norms:")
    universal_norms = [n for n in reasoner.norms.values() if n.universality > 0.8]
    for norm in universal_norms[:3]:
        print(f"   ✓ {norm.description} ({norm.norm_type.value})")
        print(f"     Importance: {norm.importance:.2f}, Context: {norm.context}")

    # Learn new norms
    print("\n2. Learning new social norms...")
    new_norm = reasoner.learn_norm(
        description="Raise hand before speaking in meetings",
        norm_type=NormType.CONVENTIONAL,
        context="formal meetings",
        importance=0.6,
        examples=["business meetings", "classroom"],
        source="observation"
    )
    print(f"   ✓ Learned new norm: {new_norm.description}")

    # Create social context
    print("\n3. Creating social context...")
    context = reasoner.create_context(
        description="Team meeting to discuss project",
        participants=["alice_001", "bob_001", "charlie_001"],
        formality=0.7
    )

    reasoner.assign_roles(context.context_id, {
        "alice_001": SocialRole.LEADER,
        "bob_001": SocialRole.PARTICIPANT,
        "charlie_001": SocialRole.PARTICIPANT
    })

    print(f"   ✓ Created context: {context.description}")
    print(f"   ✓ Active norms: {len(context.norms_active)}")
    print(f"   ✓ Participants: {len(context.participants)}")

    # Recognize intent from behavior
    print("\n4. Recognizing intent from behavior...")
    intents = reasoner.recognize_intent(
        behavior="Alice raises hand and speaks",
        actor="alice_001",
        context={"setting": "meeting"}
    )

    for intent in intents[:2]:
        print(f"   ✓ Intent: {intent.description}")
        print(f"     Type: {intent.intent_type}, Confidence: {intent.confidence:.2f}")

    # Assess appropriateness
    print("\n5. Assessing social appropriateness...")

    appropriate_behavior = "Alice helps Bob with the task"
    score1, reasons1 = reasoner.assess_appropriateness(appropriate_behavior, context)
    print(f"   ✓ Behavior: '{appropriate_behavior}'")
    print(f"     Appropriateness: {score1:.2f}")
    print(f"     Reasons: {reasons1}")

    inappropriate_behavior = "Charlie interrupts Alice repeatedly"
    score2, reasons2 = reasoner.assess_appropriateness(inappropriate_behavior, context)
    print(f"   ✓ Behavior: '{inappropriate_behavior}'")
    print(f"     Appropriateness: {score2:.2f}")
    print(f"     Reasons: {reasons2}")

    # Predict social outcomes
    print("\n6. Predicting social outcomes...")
    outcome = reasoner.predict_social_outcome(
        action="Bob suggests a different approach",
        actor="bob_001",
        context=context
    )

    print(f"   ✓ Action: {outcome['action']}")
    print(f"     Overall outcome: {outcome['overall_outcome']}")
    print(f"     Predicted reactions:")
    for reaction in outcome['predicted_reactions'][:2]:
        print(f"       - {reaction['participant']}: {reaction['reaction']} ({reaction['probability']:.2f})")

    # Record behaviors
    print("\n7. Recording and tracking behaviors...")
    reasoner.record_behavior(
        actor="alice_001",
        action="Share resources with team",
        behavior_type=BehaviorType.COOPERATIVE,
        context=context
    )

    stats = reasoner.get_statistics()
    print(f"   ✓ Total behaviors recorded: {stats['behaviors_recorded']}")
    print(f"   ✓ Total norms: {stats['total_norms']}")
    print(f"   ✓ Norm violations: {stats['norm_violations']}")

    print("\n   ✅ Social Reasoning demonstration complete!")
    return reasoner


def demo_collaboration():
    """Demonstrate Collaboration capabilities"""
    print("\n" + "=" * 80)
    print("COLLABORATION ENGINE DEMONSTRATION")
    print("=" * 80)

    collab = CollaborationEngine()

    # Create collaborative task
    print("\n1. Creating collaborative task...")
    task = collab.create_collaborative_task(
        description="Build a robot prototype",
        goal="Complete functional robot prototype",
        participants=["alice_001", "bob_001", "charlie_001"],
        coordinator="alice_001"
    )

    print(f"   ✓ Task created: {task.description}")
    print(f"   ✓ Participants: {len(task.participants)}")
    print(f"   ✓ Coordinator: {task.coordinator}")

    # Decompose task
    print("\n2. Decomposing task into subtasks...")
    subtasks = collab.decompose_task(task, strategy="specialized")

    print(f"   ✓ Created {len(subtasks)} subtasks:")
    for subtask in subtasks:
        participant = subtask.participants[0] if subtask.participants else "None"
        role = subtask.role_assignments.get(participant, "unknown")
        print(f"     - {subtask.description}")
        print(f"       Assigned to: {participant} (role: {role})")

    # Plan communication
    print("\n3. Planning communication strategy...")
    comm_strategy = collab.plan_communication(
        sender="alice_001",
        recipient="bob_001",
        purpose="Request status update on your part",
        context={"formality": 0.5, "urgency": 0.7}
    )

    print(f"   ✓ Communication strategy:")
    print(f"     Type: {comm_strategy.message_type.value}")
    print(f"     Priority: {comm_strategy.priority:.2f}")
    print(f"     Urgency: {comm_strategy.urgency:.2f}")

    # Send messages
    print("\n4. Sending and responding to messages...")
    msg = collab.send_message(
        sender="alice_001",
        recipient="bob_001",
        message_type=CommunicationType.REQUEST,
        content="Can you share your progress on the motor assembly?",
        metadata={"task_id": task.task_id}
    )

    response = collab.respond_to_message(
        message=msg,
        response_content="Motor assembly is 75% complete, on track for deadline",
        accept=True
    )

    print(f"   ✓ Message sent: {msg.message_type.value}")
    print(f"   ✓ Response received: {response.message_type.value}")
    print(f"   ✓ Response content: {response.content}")

    # Allocate resources
    print("\n5. Allocating shared resources...")
    available = {
        "compute": 100.0,
        "workspace": 50.0,
        "tools": 30.0
    }

    allocation = collab.allocate_resources(task, available)

    print(f"   ✓ Resources allocated:")
    for participant, resources in allocation.items():
        print(f"     {participant}:")
        for resource, amount in resources.items():
            print(f"       - {resource}: {amount:.1f}")

    # Detect and resolve conflict
    print("\n6. Detecting and resolving conflicts...")
    conflict = collab.detect_conflict(
        agent1="bob_001",
        agent2="charlie_001",
        context={
            "resource": "tools",
            "bob_001_request": 20.0,
            "charlie_001_request": 25.0
        }
    )

    if conflict:
        print(f"   ✓ Conflict detected:")
        print(f"     Type: {conflict.conflict_type.value}")
        print(f"     Parties: {', '.join(conflict.parties)}")
        print(f"     Description: {conflict.description}")
        print(f"     Strategy: {conflict.strategy.value}")

        resolved = collab.resolve_conflict(conflict)
        print(f"   ✓ Resolution attempted: {'Success' if resolved else 'Failed'}")
        if conflict.proposed_solution:
            print(f"     Solution: {conflict.proposed_solution}")

    # Coordinate execution
    print("\n7. Coordinating task execution...")

    # Start task
    status1 = collab.coordinate_execution(task)
    print(f"   ✓ {status1['message']}")

    # Simulate progress
    task.status = collab.tasks[task.task_id].status = TaskStatus.IN_PROGRESS
    if task.subtasks:
        # Mark first subtask complete
        first_subtask = collab.tasks[task.subtasks[0]]
        first_subtask.status = TaskStatus.COMPLETED

    status2 = collab.coordinate_execution(task)
    print(f"   ✓ Progress update: {status2['message']}")

    # Get metrics
    print("\n8. Collaboration metrics...")
    metrics = collab.get_collaboration_metrics()

    print(f"   ✓ Total tasks: {metrics['total_tasks']}")
    print(f"   ✓ Completed tasks: {metrics['completed_tasks']}")
    print(f"   ✓ Total messages: {metrics['total_messages']}")
    print(f"   ✓ Total conflicts: {metrics['total_conflicts']}")
    print(f"   ✓ Resolution rate: {metrics['resolution_rate']:.2%}")
    print(f"   ✓ Active resources: {metrics['active_resources']}")

    print("\n   ✅ Collaboration demonstration complete!")
    return collab


def demo_integrated_scenario():
    """Demonstrate integration of all three components"""
    print("\n" + "=" * 80)
    print("INTEGRATED SCENARIO: Multi-Agent Collaboration with Social Intelligence")
    print("=" * 80)

    tom = TheoryOfMind()
    reasoner = SocialReasoner()
    collab = CollaborationEngine()

    print("\nScenario: Three agents must collaborate to solve a complex problem")
    print("-" * 80)

    # Setup agents
    print("\n1. Setting up agents with mental models...")
    for agent_id, name in [("agent_1", "Alice"), ("agent_2", "Bob"), ("agent_3", "Carol")]:
        tom.register_agent(agent_id, name, capabilities={"reasoning", "communication"})
        print(f"   ✓ {name} registered")

    # Create social context
    print("\n2. Establishing social context...")
    context = reasoner.create_context(
        description="Collaborative problem-solving session",
        participants=["agent_1", "agent_2", "agent_3"],
        formality=0.5
    )

    reasoner.assign_roles(context.context_id, {
        "agent_1": SocialRole.LEADER,
        "agent_2": SocialRole.PARTICIPANT,
        "agent_3": SocialRole.PARTICIPANT
    })
    print(f"   ✓ Social context established with {len(context.participants)} participants")

    # Create collaborative task
    print("\n3. Creating collaborative task...")
    task = collab.create_collaborative_task(
        description="Solve optimization problem",
        goal="Find optimal solution within constraints",
        participants=["agent_1", "agent_2", "agent_3"],
        coordinator="agent_1"
    )
    print(f"   ✓ Task: {task.description}")

    # Theory of Mind: Observe and predict
    print("\n4. Using Theory of Mind to predict agent behaviors...")
    tom.observe_behavior(
        "agent_2",
        "Propose alternative approach",
        {"topic": "optimization method"}
    )

    predictions = tom.predict_behavior("agent_2", {"task": "optimization"})
    print(f"   ✓ Predicted behaviors for Bob:")
    for behavior, prob in predictions[:2]:
        print(f"     - {behavior} ({prob:.2f})")

    # Social Reasoning: Assess appropriateness
    print("\n5. Assessing social appropriateness of actions...")
    score, reasons = reasoner.assess_appropriateness(
        "Bob suggests trying a different algorithm",
        context
    )
    print(f"   ✓ Appropriateness score: {score:.2f}")
    if reasons:
        print(f"     Reasons: {', '.join(reasons[:2])}")

    # Collaboration: Communication and coordination
    print("\n6. Facilitating communication...")
    msg = collab.send_message(
        sender="agent_1",
        recipient="agent_2",
        message_type=CommunicationType.QUERY,
        content="What approach do you recommend?"
    )

    response = collab.respond_to_message(
        msg,
        "I recommend trying gradient descent with momentum",
        accept=True
    )
    print(f"   ✓ Communication exchange completed")

    # Update mental models from communication
    tom.update_from_communication(
        "agent_2",
        "I believe gradient descent will converge faster",
        {"topic": "optimization"}
    )
    print(f"   ✓ Mental models updated from communication")

    # Predict social outcome
    print("\n7. Predicting social outcome of collaborative approach...")
    outcome = reasoner.predict_social_outcome(
        "Team adopts Bob's suggested approach",
        "agent_1",
        context
    )
    print(f"   ✓ Predicted outcome: {outcome['overall_outcome']}")
    print(f"   ✓ Team reactions:")
    for reaction in outcome['predicted_reactions'][:2]:
        print(f"     - {reaction['participant']}: {reaction['emotion']}")

    # Execute collaboration
    print("\n8. Coordinating collaborative execution...")
    execution_status = collab.coordinate_execution(task)
    print(f"   ✓ {execution_status['message']}")

    print("\n" + "=" * 80)
    print("✅ INTEGRATED SCENARIO COMPLETE!")
    print("=" * 80)

    print("\nKey Achievements:")
    print("  • Theory of Mind: Built mental models and predicted agent behaviors")
    print("  • Social Reasoning: Assessed appropriateness and predicted outcomes")
    print("  • Collaboration: Coordinated multi-agent task execution")
    print("  • Integration: All systems working together seamlessly")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" PILLAR 9: SOCIAL INTELLIGENCE & THEORY OF MIND")
    print(" Comprehensive Demonstration")
    print("=" * 80)

    # Run all demonstrations
    tom = demo_theory_of_mind()
    reasoner = demo_social_reasoning()
    collab = demo_collaboration()

    # Integrated scenario
    demo_integrated_scenario()

    print("\n" + "=" * 80)
    print("🎉 ALL DEMONSTRATIONS COMPLETE!")
    print("=" * 80)
    print("\nPillar 9 successfully demonstrates:")
    print("  ✓ Theory of Mind - Mental state modeling and perspective taking")
    print("  ✓ Social Reasoning - Norm understanding and behavior prediction")
    print("  ✓ Collaboration - Cooperative planning and conflict resolution")
    print("  ✓ Full Integration - All components working together")
    print("\nSocial Intelligence is ready for AGI deployment!")
    print("=" * 80 + "\n")
