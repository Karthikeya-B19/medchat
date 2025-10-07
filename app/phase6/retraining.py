"""Utilities to craft retraining manifests from logged conversations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from app.phase5.logging_utils import StructuredConversationLogger


@dataclass
class RetrainingPlan:
    batch_name: str
    source_files: List[str]
    notes: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "batch_name": self.batch_name,
            "source_files": self.source_files,
            "notes": self.notes,
        }


class RetrainingScheduler:
    """Simple planner that inspects review feedback to build training batches."""

    def __init__(self, log_dir: Path = Path("app/phase5/logs")) -> None:
        self.log_dir = Path(log_dir)
        self.logger = StructuredConversationLogger(self.log_dir)

    def load_feedback(self) -> List[Dict[str, object]]:
        feedback_path = self.logger.feedback_path
        if not feedback_path.exists():
            return []
        entries: List[Dict[str, object]] = []
        with feedback_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                try:
                    entries.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
        return entries

    def generate_plan(self) -> Dict[str, object]:
        feedback = self.load_feedback()
        unsafe = [entry for entry in feedback if entry.get("resolution") == "unsafe"]
        follow_up = [entry for entry in feedback if entry.get("resolution") == "needs_follow_up"]

        batches: List[RetrainingPlan] = []
        if unsafe:
            batches.append(
                RetrainingPlan(
                    batch_name="safety-hotfix",
                    source_files=[f.get("conversation_file", "") for f in unsafe if f.get("conversation_file")],
                    notes="Prioritise unsafe conversations for guardrail fine-tuning.",
                )
            )
        if follow_up:
            batches.append(
                RetrainingPlan(
                    batch_name="coaching-review",
                    source_files=[f.get("conversation_file", "") for f in follow_up if f.get("conversation_file")],
                    notes="Review clinician feedback and incorporate clarifying questions.",
                )
            )

        schedule = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "log_dir": str(self.log_dir),
            "planned_batches": [batch.to_dict() for batch in batches],
            "feedback_samples": feedback[:5],
        }
        return schedule


if __name__ == "__main__":
    scheduler = RetrainingScheduler()
    plan = scheduler.generate_plan()
    print(json.dumps(plan, indent=2))
