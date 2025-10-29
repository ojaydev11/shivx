"""
Neural Pattern Recorder
Records all computational patterns during AGI experiments (like brain recording)
"""
import sqlite3
import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import numpy as np

from .schemas import NeuralPattern, AGIApproachType


class PatternRecorder:
    """Records and retrieves neural patterns from experiments"""

    def __init__(self, db_path: str = "data/agi_lab/patterns.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize pattern database"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                approach_type TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                context TEXT,
                data TEXT,  -- JSON serialized
                timestamp TEXT,
                success_score REAL,
                generalization_score REAL,
                novelty_score REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_approach_success
            ON patterns(approach_type, success_score DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern_type
            ON patterns(pattern_type, timestamp DESC)
        """)
        conn.commit()
        conn.close()

    def record_pattern(self, pattern: NeuralPattern) -> None:
        """Record a neural pattern"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            INSERT OR REPLACE INTO patterns
            (pattern_id, approach_type, pattern_type, context, data, timestamp,
             success_score, generalization_score, novelty_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pattern.pattern_id,
            pattern.approach_type.value,
            pattern.pattern_type,
            pattern.context,
            json.dumps(pattern.data),
            pattern.timestamp.isoformat(),
            pattern.success_score,
            pattern.generalization_score,
            pattern.novelty_score,
        ))
        conn.commit()
        conn.close()

    def get_best_patterns(
        self,
        approach_type: Optional[AGIApproachType] = None,
        min_success: float = 0.8,
        limit: int = 100
    ) -> List[NeuralPattern]:
        """Retrieve best performing patterns"""
        conn = sqlite3.connect(str(self.db_path))

        if approach_type:
            query = """
                SELECT * FROM patterns
                WHERE approach_type = ? AND success_score >= ?
                ORDER BY success_score DESC, generalization_score DESC
                LIMIT ?
            """
            rows = conn.execute(query, (approach_type.value, min_success, limit)).fetchall()
        else:
            query = """
                SELECT * FROM patterns
                WHERE success_score >= ?
                ORDER BY success_score DESC, generalization_score DESC
                LIMIT ?
            """
            rows = conn.execute(query, (min_success, limit)).fetchall()

        conn.close()

        patterns = []
        for row in rows:
            patterns.append(NeuralPattern(
                pattern_id=row[0],
                approach_type=AGIApproachType(row[1]),
                pattern_type=row[2],
                context=row[3],
                data=json.loads(row[4]),
                timestamp=datetime.fromisoformat(row[5]),
                success_score=row[6],
                generalization_score=row[7],
                novelty_score=row[8],
            ))

        return patterns

    def get_similar_patterns(self, context: str, k: int = 10) -> List[NeuralPattern]:
        """Find patterns from similar contexts (simple text match)"""
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("""
            SELECT * FROM patterns
            WHERE context LIKE ?
            ORDER BY success_score DESC
            LIMIT ?
        """, (f"%{context}%", k)).fetchall()
        conn.close()

        patterns = []
        for row in rows:
            patterns.append(NeuralPattern(
                pattern_id=row[0],
                approach_type=AGIApproachType(row[1]),
                pattern_type=row[2],
                context=row[3],
                data=json.loads(row[4]),
                timestamp=datetime.fromisoformat(row[5]),
                success_score=row[6],
                generalization_score=row[7],
                novelty_score=row[8],
            ))

        return patterns

    def analyze_approach(self, approach_type: AGIApproachType) -> Dict[str, Any]:
        """Analyze performance of an approach across all patterns"""
        conn = sqlite3.connect(str(self.db_path))

        stats = conn.execute("""
            SELECT
                COUNT(*) as count,
                AVG(success_score) as avg_success,
                AVG(generalization_score) as avg_generalization,
                AVG(novelty_score) as avg_novelty,
                MAX(success_score) as max_success
            FROM patterns
            WHERE approach_type = ?
        """, (approach_type.value,)).fetchone()

        conn.close()

        return {
            "approach": approach_type.value,
            "total_patterns": stats[0],
            "avg_success": stats[1] or 0.0,
            "avg_generalization": stats[2] or 0.0,
            "avg_novelty": stats[3] or 0.0,
            "max_success": stats[4] or 0.0,
        }

    def consolidate_patterns(self, min_similarity: float = 0.85) -> int:
        """Consolidate similar patterns (brain memory consolidation)"""
        # Simple version: merge patterns with same context and high scores
        conn = sqlite3.connect(str(self.db_path))

        # Group by context
        groups = conn.execute("""
            SELECT context, COUNT(*) as cnt
            FROM patterns
            GROUP BY context
            HAVING cnt > 1
        """).fetchall()

        merged = 0
        for context, count in groups:
            # Get all patterns for this context
            patterns = conn.execute("""
                SELECT pattern_id, success_score, generalization_score
                FROM patterns
                WHERE context = ?
                ORDER BY success_score DESC
            """, (context,)).fetchall()

            if len(patterns) > 1:
                # Keep the best, delete others
                best_id = patterns[0][0]
                for pid, _, _ in patterns[1:]:
                    conn.execute("DELETE FROM patterns WHERE pattern_id = ?", (pid,))
                    merged += 1

        conn.commit()
        conn.close()

        return merged
