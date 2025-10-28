#!/usr/bin/env python3
"""
Performance test data generator for SLMG.

Generates 50k+ nodes with realistic distribution:
- Episodic events with temporal chains
- Semantic facts with entity relationships
- Procedural skills with dependencies
- Code snippets
"""

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.api import MemoryAPI


def generate_episodic_events(api: MemoryAPI, count: int, progress_cb=None):
    """Generate episodic events."""
    participants = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
    locations = ["Conference Room", "Office", "Lab", "Cafeteria", "Remote"]
    activities = [
        "brainstorming session",
        "code review",
        "standup meeting",
        "demo presentation",
        "team lunch",
        "1-on-1 sync",
        "sprint planning",
        "retrospective",
    ]

    start_date = datetime.utcnow() - timedelta(days=365)

    for i in range(count):
        timestamp = start_date + timedelta(minutes=i * 30)
        activity = random.choice(activities)
        parts = random.sample(participants, k=random.randint(2, 4))
        location = random.choice(locations)

        description = f"Had a {activity} with {', '.join(parts[:2])}"
        if len(parts) > 2:
            description += f" and {len(parts)-2} others"

        api.store_event(
            description=description,
            timestamp=timestamp,
            participants=parts,
            location=location,
            importance=random.uniform(0.3, 0.9),
            tags=["work", activity.split()[0]],
        )

        if progress_cb and i % 100 == 0:
            progress_cb(i, count, "events")


def generate_semantic_facts(api: MemoryAPI, count: int, progress_cb=None):
    """Generate semantic facts."""
    entities = [
        "ShivX", "Python", "Machine Learning", "AGI", "Memory System",
        "Trading", "Solana", "FastAPI", "PostgreSQL", "Redis",
    ]
    predicates = [
        "uses", "implements", "supports", "requires", "integrates with",
        "is a", "enables", "optimizes", "provides", "manages",
    ]
    objects = [
        "local processing", "privacy protection", "continuous learning",
        "semantic memory", "spatial reasoning", "self-repair",
        "database", "API", "microservices", "containers",
    ]

    for i in range(count):
        subject = random.choice(entities)
        predicate = random.choice(predicates)
        obj = random.choice(objects)

        api.store_fact(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=random.uniform(0.7, 1.0),
            importance=random.uniform(0.5, 0.9),
        )

        if progress_cb and i % 100 == 0:
            progress_cb(i, count, "facts")


def generate_procedural_skills(api: MemoryAPI, count: int, progress_cb=None):
    """Generate procedural skills."""
    skill_templates = [
        {
            "name": "deploy_model",
            "description": "Deploy ML model to production",
            "steps": [
                "Run test suite",
                "Build Docker image",
                "Push to registry",
                "Update deployment manifest",
                "Apply to cluster",
                "Verify health checks",
            ],
        },
        {
            "name": "debug_performance",
            "description": "Debug performance issues",
            "steps": [
                "Collect profiling data",
                "Identify bottlenecks",
                "Analyze query patterns",
                "Optimize hot paths",
                "Measure improvements",
            ],
        },
        {
            "name": "onboard_developer",
            "description": "Onboard new team member",
            "steps": [
                "Set up development environment",
                "Grant access to systems",
                "Walk through codebase",
                "Assign first task",
                "Schedule check-ins",
            ],
        },
    ]

    for i in range(count):
        template = random.choice(skill_templates)
        variant = f"{template['name']}_{i}"

        api.store_skill(
            name=variant,
            description=template["description"],
            steps=template["steps"],
            importance=random.uniform(0.6, 0.9),
            tags=["procedure", "skill"],
        )

        if progress_cb and i % 50 == 0:
            progress_cb(i, count, "skills")


def generate_code_snippets(api: MemoryAPI, count: int, progress_cb=None):
    """Generate code snippets."""
    code_templates = [
        ("async def fetch_data(url: str) -> dict:\n    async with httpx.AsyncClient() as client:\n        response = await client.get(url)\n        return response.json()", "Async HTTP fetch"),
        ("def calculate_metrics(data: pd.DataFrame) -> dict:\n    return {\n        'mean': data.mean(),\n        'std': data.std(),\n        'count': len(data),\n    }", "Calculate statistics"),
        ("with Session() as session:\n    result = session.query(Model).filter(Model.id == id).first()\n    if result:\n        session.delete(result)\n        session.commit()", "Database delete operation"),
    ]

    for i in range(count):
        code, desc = random.choice(code_templates)
        variant_desc = f"{desc} (variant {i})"

        api.store_code(
            code=code,
            description=variant_desc,
            language="python",
            tags=["code", "example"],
        )

        if progress_cb and i % 50 == 0:
            progress_cb(i, count, "code")


def main():
    parser = argparse.ArgumentParser(description="Generate performance test data")
    parser.add_argument("--target", type=int, default=50000, help="Target node count")
    parser.add_argument("--db-path", default="./data/memory/perf_test.db", help="Database path")
    parser.add_argument("--quick", action="store_true", help="Quick mode (10k nodes)")
    args = parser.parse_args()

    target = 10000 if args.quick else args.target

    print(f"🏗️  Generating {target:,} memory nodes for performance testing...")
    print(f"   Database: {args.db_path}\n")

    # Distribution: 60% events, 25% facts, 10% skills, 5% code
    event_count = int(target * 0.60)
    fact_count = int(target * 0.25)
    skill_count = int(target * 0.10)
    code_count = int(target * 0.05)

    def progress(current, total, type_name):
        pct = (current / total) * 100
        print(f"   {type_name}: {current:,}/{total:,} ({pct:.1f}%)", end="\r")

    api = MemoryAPI(db_path=args.db_path, device="cpu")

    try:
        print(f"📅 Generating {event_count:,} episodic events...")
        generate_episodic_events(api, event_count, progress)
        print(f"\n   ✅ {event_count:,} events created")

        print(f"\n💡 Generating {fact_count:,} semantic facts...")
        generate_semantic_facts(api, fact_count, progress)
        print(f"\n   ✅ {fact_count:,} facts created")

        print(f"\n🔧 Generating {skill_count:,} procedural skills...")
        generate_procedural_skills(api, skill_count, progress)
        print(f"\n   ✅ {skill_count:,} skills created")

        print(f"\n💻 Generating {code_count:,} code snippets...")
        generate_code_snippets(api, code_count, progress)
        print(f"\n   ✅ {code_count:,} code snippets created")

        # Get final stats
        stats = api.get_stats()
        print(f"\n" + "=" * 60)
        print("✅ Generation Complete!")
        print("=" * 60)
        print(f"Total nodes: {stats['total_nodes']:,}")
        print(f"Total edges: {stats['total_edges']:,}")
        print(f"Database: {args.db_path}")
        print("=" * 60)

    finally:
        api.close()


if __name__ == "__main__":
    main()
