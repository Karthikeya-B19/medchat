"""Configuration objects for adaptive response categories used by the MedGuide engine.

The goal of this module is to make the response-planning logic declarative: each
intent category describes its focus area, instructions, and default behaviour.
This keeps the core engine slimmer and makes it easier to audit or tweak the
NLG strategy without editing large conditional blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ResponseCategory:
    """Declarative configuration for a single response category."""

    name: str
    response_focus: str
    resource_type: str
    explanation_instruction: str
    safety_instruction: str
    support_instruction: str
    detail_bias: str = "balanced"
    response_length: str = "detailed"
    default_follow_ups: List[str] = field(default_factory=list)
    escalate_on_match: bool = False


GENERAL_FOLLOW_UP = [
    "When did each symptom begin?",
    "What makes the symptoms better or worse?",
    "Do you have chest pain, shortness of breath, or difficulty swallowing?",
]


CATEGORY_REGISTRY: Dict[str, ResponseCategory] = {
    "urgent_symptom": ResponseCategory(
        name="Urgent symptom",
        response_focus="urgent_physical",
        resource_type="emergency",
        explanation_instruction="Prioritise critical differentials tied to the red flag symptom and stress the seriousness.",
        safety_instruction="Direct the patient to immediate emergency services and outline warning signs to monitor while help arrives.",
        support_instruction="Stay calm, validate fear, and keep the patient focused on practical safety steps until clinicians take over.",
        detail_bias="clinical",
        response_length="detailed",
        escalate_on_match=True,
    ),
    "greeting": ResponseCategory(
        name="Greeting",
        response_focus="greeting",
        resource_type="greeting",
        explanation_instruction="Offer a welcoming orientation describing the assistant's role and what information is helpful.",
        safety_instruction="Invite the patient to report symptoms or urgent concerns so triage can begin if needed.",
        support_instruction="Encourage open sharing and reassure the patient that you will guide them through the process.",
        detail_bias="balanced",
        response_length="brief",
        default_follow_ups=[
            "What symptoms are you experiencing right now?",
            *GENERAL_FOLLOW_UP,
        ],
    ),
    "symptom_report": ResponseCategory(
        name="Initial symptom report",
        response_focus="physical_monitoring",
        resource_type="medical",
        explanation_instruction="Summarise likely causes based on the new symptoms and state what information narrows the assessment.",
        safety_instruction="Share tiered follow-up guidance covering self-care, clinician contact, and emergency escalation.",
        support_instruction="Validate discomfort and encourage monitoring along with prompt updates if symptoms worsen.",
        detail_bias="balanced",
        response_length="detailed",
        default_follow_ups=GENERAL_FOLLOW_UP,
    ),
    "symptom_update": ResponseCategory(
        name="Symptom update",
        response_focus="symptom_update",
        resource_type="medical",
        explanation_instruction="Integrate the new symptom details with existing history to adjust the working assessment.",
        safety_instruction="Explain how monitoring or escalation guidance changes with the updated symptom picture.",
        support_instruction="Acknowledge the effort to report updates and describe how that helps tailor next steps.",
        detail_bias="balanced",
        response_length="detailed",
        default_follow_ups=GENERAL_FOLLOW_UP,
    ),
    "follow_up": ResponseCategory(
        name="Follow up",
        response_focus="follow_up",
        resource_type="follow_up",
        explanation_instruction="Review known symptoms and clarify what remains stable versus what needs attention.",
        safety_instruction="Reiterate the warning signs that require clinician contact or emergency care.",
        support_instruction="Appreciate the check-in and encourage consistent symptom tracking.",
        detail_bias="balanced",
        response_length="brief",
        default_follow_ups=[
            "How have your symptoms changed since the last update?",
            *GENERAL_FOLLOW_UP[:2],
        ],
    ),
    "symptom_reinforce": ResponseCategory(
        name="Symptom reinforcement",
        response_focus="physical_monitoring",
        resource_type="medical",
        explanation_instruction="Restate the symptoms being tracked and ensure the patient understands the safety plan.",
        safety_instruction="Specify which changes in severity or frequency should trigger medical review.",
        support_instruction="Encourage ongoing tracking and open communication if new concerns emerge.",
        detail_bias="balanced",
        response_length="brief",
        default_follow_ups=GENERAL_FOLLOW_UP,
    ),
    "mixed": ResponseCategory(
        name="Mixed physical + emotional",
        response_focus="mixed",
        resource_type="integrated",
        explanation_instruction="Address physical risks while acknowledging how stress can intensify symptom perception.",
        safety_instruction="Outline both medical escalation and when to seek mental health support.",
        support_instruction="Balance reassurance with coping strategies to reduce anxiety-driven symptom amplification.",
        detail_bias="balanced",
        response_length="detailed",
        default_follow_ups=GENERAL_FOLLOW_UP,
    ),
    "emotional_support": ResponseCategory(
        name="Emotional support",
        response_focus="emotional_support",
        resource_type="mental_health",
        explanation_instruction="Clarify that no new danger signs were found and focus on emotional regulation strategies.",
        safety_instruction="Encourage the patient to seek help if distress escalates or suicidal thoughts appear.",
        support_instruction="Offer grounding techniques and normalise the emotional response.",
        detail_bias="simple",
        response_length="detailed",
        default_follow_ups=[
            "What tends to help you feel a little safer when anxiety rises?",
            "Do you have someone you can reach out to for support right now?",
        ],
    ),
    "general": ResponseCategory(
        name="General guidance",
        response_focus="general",
        resource_type="general",
        explanation_instruction="Provide a balanced summary and invite more detail so advice can be tailored.",
        safety_instruction="Remind the patient of core monitoring steps and when to contact a professional.",
        support_instruction="Keep the tone compassionate and encourage ongoing dialogue.",
        detail_bias="balanced",
        response_length="detailed",
        default_follow_ups=GENERAL_FOLLOW_UP,
    ),
}

DEFAULT_CATEGORY_KEY = "general"


def get_category(intent: str, urgent: bool) -> ResponseCategory:
    """Return the category config for an intent and urgency flag."""

    if urgent:
        return CATEGORY_REGISTRY["urgent_symptom"]
    if intent in CATEGORY_REGISTRY:
        return CATEGORY_REGISTRY[intent]
    return CATEGORY_REGISTRY[DEFAULT_CATEGORY_KEY]


__all__ = ["ResponseCategory", "CATEGORY_REGISTRY", "get_category"]
