"""Structured logging and active-learning utilities for the MedGuide chatbot."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

UTC = timezone.utc


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _default_serializer(obj):  # pragma: no cover - safety guard
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


class StructuredConversationLogger:
    """Persist conversations with metadata and surface review candidates."""

    schema_version = "2025-09-28"

    def __init__(self, log_dir: Path, review_filename: str = "review_queue.jsonl") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.review_path = self.log_dir / review_filename
        self.metrics_path = self.log_dir / "analytics.json"
        self.feedback_path = self.log_dir / "review_feedback.jsonl"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def persist_conversation(
        self,
        conversation: List[Dict[str, object]],
        metadata_log: List[Dict[str, object]],
        patient_profile,
    ) -> Path:
        timestamp = datetime.now(UTC)
        filename = f"conversation_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "schema_version": self.schema_version,
            "saved_at": timestamp.isoformat(),
            "patient_profile": self._serialise_profile(patient_profile),
            "conversation": conversation,
            "metadata_log": metadata_log,
            "auto_score_summary": self._summarise_scores(metadata_log),
        }
        output_path = self.log_dir / filename
        output_path.write_text(json.dumps(payload, indent=2, default=_default_serializer), encoding="utf-8")

        self._update_metrics(payload["auto_score_summary"])
        self._maybe_enqueue_reviewItems(filename, metadata_log)
        return output_path

    def load_review_queue(self, limit: int = 20) -> List[Dict[str, object]]:
        if not self.review_path.exists():
            return []
        entries: List[Dict[str, object]] = []
        with self.review_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:  # pragma: no cover - defensive
                    continue
        entries.sort(key=lambda item: item.get("queued_at", ""), reverse=True)
        return entries[:limit]

    def mark_reviewed(
        self,
        conversation_file: str,
        turn_index: int,
        reviewer: str,
        resolution: str = "approved",
        notes: str = "",
    ) -> None:
        if not self.review_path.exists():
            return
        updated: List[str] = []
        timestamp = _iso_now()
        with self.review_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:  # pragma: no cover - defensive
                    continue
                if entry.get("conversation_file") == conversation_file and entry.get("turn_index") == turn_index:
                    entry["status"] = resolution
                    entry["reviewed_by"] = reviewer
                    entry["reviewed_at"] = timestamp
                    if notes:
                        entry["notes"] = notes
                updated.append(json.dumps(entry))
        self.review_path.write_text("\n".join(updated) + ("\n" if updated else ""), encoding="utf-8")

    def record_review_feedback(
        self,
        conversation_file: str,
        turn_index: int,
        reviewer: str,
        resolution: str,
        notes: str = "",
        updated_prompt: str = "",
    ) -> None:
        payload = {
            "submitted_at": _iso_now(),
            "conversation_file": conversation_file,
            "turn_index": turn_index,
            "reviewer": reviewer,
            "resolution": resolution,
            "notes": notes,
            "updated_prompt": updated_prompt,
        }
        with self.feedback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        self.mark_reviewed(conversation_file, turn_index, reviewer, resolution=resolution, notes=notes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _serialise_profile(self, profile) -> Dict[str, object]:
        if hasattr(profile, "__dict__"):
            return {k: v for k, v in profile.__dict__.items() if v not in (None, "")}
        try:
            return asdict(profile)
        except Exception:  # pragma: no cover - safety guard
            return {
                "name": getattr(profile, "name", None),
                "age": getattr(profile, "age", None),
                "gender": getattr(profile, "gender", None),
            }

    def _summarise_scores(self, metadata_log: Iterable[Dict[str, object]]) -> Dict[str, float]:
        aggregates: Dict[str, List[float]] = {}
        for entry in metadata_log:
            scores = entry.get("auto_scores") or {}
            for key, value in scores.items():
                try:
                    aggregates.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    continue
        return {key: (sum(values) / len(values) if values else 0.0) for key, values in aggregates.items()}

    def _update_metrics(self, summary: Dict[str, float]) -> None:
        metrics = {"updated_at": _iso_now(), "runs": []}
        if self.metrics_path.exists():
            try:
                metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:  # pragma: no cover - defensive
                metrics = {"updated_at": _iso_now(), "runs": []}
        metrics.setdefault("runs", []).append({"timestamp": _iso_now(), "scores": summary})
        metrics["updated_at"] = _iso_now()
        self.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def _maybe_enqueue_reviewItems(self, conversation_file: str, metadata_log: Iterable[Dict[str, object]]) -> None:
        review_lines: List[str] = []
        for index, entry in enumerate(metadata_log):
            if not entry.get("needs_review"):
                continue
            review_lines.append(
                json.dumps(
                    {
                        "queued_at": _iso_now(),
                        "conversation_file": conversation_file,
                        "turn_index": index,
                        "review_reasons": entry.get("review_reasons", []),
                        "auto_scores": entry.get("auto_scores", {}),
                        "intent": entry.get("intent"),
                        "response_focus": entry.get("response_focus"),
                        "progressive_summary": entry.get("progressive_summary"),
                    }
                )
            )
        if review_lines:
            with self.review_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(review_lines) + "\n")