#!/usr/bin/env python3
"""
Semantic Long-Term Memory Graph (SLMG) Demo

Demonstrates:
1. Storing episodic events
2. Storing semantic facts
3. Storing procedural skills
4. Hybrid retrieval
5. Memory consolidation
6. Daemon operation
"""

import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from daemons.memory_daemon import MemoryDaemon
from memory.api import MemoryAPI
from memory.schemas import MemoryMode

console = Console()


def demo_episodic_memory(api: MemoryAPI):
    """Demonstrate episodic memory capabilities."""
    console.print("\n[bold cyan]1. EPISODIC MEMORY DEMO[/bold cyan]")
    console.print("Storing time-stamped events and experiences...\n")

    # Store various events
    events = [
        {
            "description": "Had a brainstorming session about the new AGI architecture",
            "participants": ["Alice", "Bob", "Charlie"],
            "location": "Conference Room",
            "outcome": "Decided to implement memory graph system",
            "importance": 0.9,
        },
        {
            "description": "Completed implementation of graph store with SQLite and FTS5",
            "participants": ["Dev Team"],
            "outcome": "Core storage layer working",
            "importance": 0.8,
        },
        {
            "description": "Coffee break and casual discussion about ML trends",
            "participants": ["Alice", "Charlie"],
            "location": "Cafeteria",
            "importance": 0.3,
        },
    ]

    for event in events:
        event_id = api.store_event(**event)
        console.print(f"✓ Stored event: [green]{event['description'][:50]}...[/green]")

    # Recall recent events
    console.print("\n[yellow]Recalling recent events...[/yellow]")
    recent = api.recall_recent_events(days=7, limit=5)

    table = Table(title="Recent Events", show_header=True)
    table.add_column("Time", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Importance", style="green")

    for event in recent:
        table.add_row(
            event.created_at.strftime("%H:%M:%S"),
            event.content[:60] + "..." if len(event.content) > 60 else event.content,
            f"{event.importance:.2f}",
        )

    console.print(table)


def demo_semantic_memory(api: MemoryAPI):
    """Demonstrate semantic memory capabilities."""
    console.print("\n[bold cyan]2. SEMANTIC MEMORY DEMO[/bold cyan]")
    console.print("Storing facts and knowledge...\n")

    # Store facts about a topic
    facts = [
        ("ShivX", "is a", "privacy-first AGI platform"),
        ("ShivX", "uses", "local LLMs"),
        ("ShivX", "implements", "semantic memory graph"),
        ("Memory Graph", "uses", "SQLite with FTS5"),
        ("Memory Graph", "supports", "vector similarity search"),
    ]

    for subject, predicate, obj in facts:
        api.store_fact(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=0.95,
            importance=0.7,
        )
        console.print(f"✓ Stored fact: [green]{subject} {predicate} {obj}[/green]")

    # Recall facts about an entity
    console.print("\n[yellow]Recalling facts about 'ShivX'...[/yellow]")
    facts_about_shivx = api.recall_facts_about("ShivX", limit=10)

    for fact in facts_about_shivx:
        console.print(f"  • {fact.content} [dim](confidence: {fact.confidence:.2f})[/dim]")


def demo_procedural_memory(api: MemoryAPI):
    """Demonstrate procedural memory capabilities."""
    console.print("\n[bold cyan]3. PROCEDURAL MEMORY DEMO[/bold cyan]")
    console.print("Storing skills and procedures...\n")

    # Store a skill
    api.store_skill(
        name="train_ml_model",
        description="How to train a machine learning model",
        steps=[
            "Load and prepare dataset",
            "Split data into train/validation/test sets",
            "Initialize model architecture",
            "Define loss function and optimizer",
            "Train model with early stopping",
            "Evaluate on test set",
            "Save model checkpoint",
        ],
        prerequisites=["data_preprocessing", "model_selection"],
        importance=0.8,
        tags=["ml", "training"],
    )
    console.print("✓ Stored skill: [green]train_ml_model[/green]")

    # Store a code snippet
    api.store_code(
        code="""
def consolidate_memory(graph, encoder, threshold=0.85):
    \"\"\"Consolidate similar memories in the graph.\"\"\"
    similar_pairs = find_similar_nodes(graph, encoder, threshold)
    for node_a, node_b in similar_pairs:
        merge_nodes(graph, node_a, node_b)
    return len(similar_pairs)
""",
        description="Memory consolidation function",
        language="python",
        tags=["memory", "consolidation"],
    )
    console.print("✓ Stored code: [green]consolidate_memory function[/green]")

    # Recall the skill
    console.print("\n[yellow]Recalling skill 'train_ml_model'...[/yellow]")
    skill = api.recall_skill("train_ml_model")

    if skill:
        console.print(f"\n[bold]{skill.metadata['name']}[/bold]")
        console.print(f"[dim]{skill.metadata['description']}[/dim]\n")
        console.print("[bold]Steps:[/bold]")
        for i, step in enumerate(skill.metadata["steps"], 1):
            console.print(f"  {i}. {step}")


def demo_hybrid_retrieval(api: MemoryAPI):
    """Demonstrate hybrid retrieval capabilities."""
    console.print("\n[bold cyan]4. HYBRID RETRIEVAL DEMO[/bold cyan]")
    console.print("Combining dense, sparse, and graph-based search...\n")

    # Query with different modes
    queries = [
        ("AGI architecture planning", MemoryMode.EPISODIC),
        ("ShivX memory system", MemoryMode.SEMANTIC),
        ("how to train models", MemoryMode.PROCEDURAL),
        ("memory consolidation", MemoryMode.HYBRID),
    ]

    for query, mode in queries:
        console.print(f"\n[yellow]Query:[/yellow] '{query}' [dim](mode: {mode.value})[/dim]")
        results = api.recall(query, k=3, mode=mode)

        console.print(f"[green]Found {len(results.nodes)} results in {results.metadata['latency_ms']:.1f}ms[/green]")

        for i, node in enumerate(results.nodes, 1):
            score = results.scores[i - 1] if i <= len(results.scores) else 0
            content_preview = node.content[:80] + "..." if len(node.content) > 80 else node.content
            console.print(
                f"  {i}. [bold]{content_preview}[/bold] "
                f"[dim](score: {score:.3f}, importance: {node.importance:.2f})[/dim]"
            )


def demo_consolidation(api: MemoryAPI):
    """Demonstrate memory consolidation."""
    console.print("\n[bold cyan]5. MEMORY CONSOLIDATION DEMO[/bold cyan]")
    console.print("Running consolidation to optimize memory...\n")

    # Get initial stats
    initial_stats = api.get_stats()
    console.print(f"Initial nodes: {initial_stats['total_nodes']}")
    console.print(f"Initial edges: {initial_stats['total_edges']}\n")

    # Run consolidation
    console.print("[yellow]Running consolidation...[/yellow]")
    report = api.consolidate()

    console.print(f"\n[green]Consolidation complete![/green]")
    console.print(f"  • Nodes merged: {report.nodes_merged}")
    console.print(f"  • Nodes pruned: {report.nodes_pruned}")
    console.print(f"  • Edges strengthened: {report.edges_strengthened}")
    console.print(f"  • Edges weakened: {report.edges_weakened}")
    console.print(f"  • Duration: {report.duration_seconds:.2f}s")

    # Get final stats
    final_stats = api.get_stats()
    console.print(f"\nFinal nodes: {final_stats['total_nodes']}")
    console.print(f"Final edges: {final_stats['total_edges']}")


def demo_daemon(api: MemoryAPI):
    """Demonstrate memory daemon operation."""
    console.print("\n[bold cyan]6. MEMORY DAEMON DEMO[/bold cyan]")
    console.print("Starting background maintenance daemon...\n")

    # Create and start daemon
    daemon = MemoryDaemon(
        graph_store=api.graph_store,
        text_encoder=api.text_encoder,
        consolidation_interval_hours=1,  # 1 hour for demo
        snapshot_interval_hours=24,
    )

    console.print("[green]Starting daemon...[/green]")
    daemon.start()

    # Check status
    time.sleep(1)
    status = daemon.get_status()

    table = Table(title="Daemon Status", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Running", "✓ Yes" if status["running"] else "✗ No")
    table.add_row("Last Consolidation", str(status["last_consolidation"] or "Never"))
    table.add_row("Last Snapshot", str(status["last_snapshot"] or "Never"))
    table.add_row("Next Consolidation", str(status["next_consolidation"]))

    console.print(table)

    # Stop daemon
    console.print("\n[yellow]Stopping daemon...[/yellow]")
    daemon.stop()
    console.print("[green]Daemon stopped[/green]")


def main():
    """Run all demos."""
    console.print(Panel.fit(
        "[bold cyan]ShivX AGI - Semantic Long-Term Memory Graph Demo[/bold cyan]\n"
        "[dim]Demonstrating human-like memory capabilities[/dim]",
        border_style="cyan",
    ))

    # Initialize memory API
    console.print("\n[yellow]Initializing memory system...[/yellow]")
    demo_db_path = Path("./data/memory/demo_memory.db")
    demo_db_path.parent.mkdir(parents=True, exist_ok=True)

    api = MemoryAPI(db_path=str(demo_db_path), device="cpu")
    console.print("[green]Memory system ready![/green]")

    try:
        # Run all demos
        demo_episodic_memory(api)
        demo_semantic_memory(api)
        demo_procedural_memory(api)
        demo_hybrid_retrieval(api)
        demo_consolidation(api)
        demo_daemon(api)

        # Final stats
        console.print("\n[bold cyan]FINAL STATISTICS[/bold cyan]")
        stats = api.get_stats()
        console.print(Panel(
            f"[bold]Total Nodes:[/bold] {stats['total_nodes']}\n"
            f"[bold]Total Edges:[/bold] {stats['total_edges']}\n"
            f"[bold]Events:[/bold] {stats['node_types']['events']}\n"
            f"[bold]Facts:[/bold] {stats['node_types']['facts']}\n"
            f"[bold]Skills:[/bold] {stats['node_types']['skills']}\n"
            f"[bold]Code:[/bold] {stats['node_types']['code']}",
            title="Memory Graph Statistics",
            border_style="green",
        ))

        console.print("\n[bold green]✓ Demo complete![/bold green]")
        console.print(f"[dim]Database saved to: {demo_db_path}[/dim]")

    finally:
        api.close()


if __name__ == "__main__":
    main()
