"""Adaptive Prompting Engine module extracted from Phase 4 notebook.

This module mirrors the logic defined in `adaptive_prompting_engine.ipynb` so that
other Python components (e.g. the Streamlit prototype) can import it without
relying on notebook execution order.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .prompt_categories import ResponseCategory, get_category

UTC = timezone.utc

STRESS_KEYWORDS = {"scared", "worried", "urgent", "panic", "afraid", "terrified", "anxious"}
SIMPLE_LANGUAGE_MARKERS = {"explain simply", "in simple terms", "plain language", "layman", "easy to understand"}
CLINICIAN_MARKERS = {"as a doctor", "as a clinician", "medical detail", "technical detail", "pathophysiology"}
EMERGENCY_KEYWORDS = {
    "severe chest pain",
    "chest tightness",
    "shortness of breath",
    "stroke",
    "loss of consciousness",
    "cannot breathe",
    "trouble breathing",
    "uncontrolled bleeding",
    "suicidal",
    "high fever",
    "fever over 102",
    "fever over 103",
}
EMOTIONAL_DISTRESS_KEYWORDS = {
    "anxious",
    "anxiety",
    "overwhelmed",
    "panic",
    "terrified",
    "scared",
    "fearful",
    "can't cope",
    "stress",
    "stressed",
    "nervous",
    "worried sick",
}

GREETINGS = {"hello", "hi", "hey", "good morning", "good evening", "good afternoon"}

FOLLOW_UP_KEYWORDS = {"checking in", "follow up", "follow-up", "just wanted to"}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "to",
    "we",
    "with",
    "you",
    "your",
}


def normalise_patient_text(raw_text: str) -> Dict[str, object]:
    lowered = raw_text.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    tokens = [token for token in cleaned.split() if token and token not in STOPWORDS]
    return {"lowered": lowered, "cleaned": cleaned, "tokens": tokens}

RED_FLAG_CATEGORY_MAP = {
    "chest": (
        "Cardiac or pulmonary emergency",
        "Chest pain or tightness can signal a heart attack or lung emergency and needs ambulance-level care.",
    ),
    "shortness of breath": (
        "Respiratory distress",
        "Trouble breathing may worsen quickly—ensure immediate medical evaluation.",
    ),
    "difficulty swallowing": (
        "Airway compromise",
        "Difficulty swallowing can mean airway swelling; urgent in-person assessment is required.",
    ),
    "cannot breathe": (
        "Airway compromise",
        "Inability to breathe is a medical emergency—activate emergency services now.",
    ),
    "fever over": (
        "Systemic infection concern",
        "High fevers above 102°F warrant clinician review, especially with chronic conditions.",
    ),
    "high fever": (
        "Systemic infection concern",
        "High fever may indicate severe infection—escalate to urgent care if persistent.",
    ),
    "confusion": (
        "Neurological red flag",
        "Sudden confusion can signify brain or metabolic emergencies; involve emergency services.",
    ),
    "stiff neck": (
        "Neurological red flag",
        "Neck stiffness with headache or fever can signal meningitis—seek emergency care immediately.",
    ),
    "blurred vision": (
        "Neurological red flag",
        "Vision changes with headache may indicate neurological emergencies; escalate for urgent evaluation.",
    ),
    "vision changes": (
        "Neurological red flag",
        "Vision changes with headache may indicate neurological emergencies; escalate for urgent evaluation.",
    ),
    "stroke": (
        "Neurological red flag",
        "Stroke symptoms need rapid emergency activation—call emergency services immediately.",
    ),
    "loss of consciousness": (
        "Neurological red flag",
        "Loss of consciousness requires urgent evaluation for cardiac or neurological causes.",
    ),
    "uncontrolled bleeding": (
        "Hemorrhage risk",
        "Severe bleeding can be life-threatening—seek emergency care without delay.",
    ),
    "suicidal": (
        "Mental health crisis",
        "Active suicidal thoughts demand immediate crisis intervention and emergency support.",
    ),
}

SYMPTOM_FOLLOW_UP_PROMPTS = {
    "fever": [
        "When did the fever start and what was the highest temperature recorded?",
        "Have you taken any fever-reducing medications, and did they help?",
        "Are you noticing chills, sweating, or rash along with the fever?",
    ],
    "sore throat": [
        "Is swallowing painful or difficult?",
        "Have you noticed swelling, white patches, or changes in your voice?",
        "Are you able to stay hydrated despite the throat discomfort?",
    ],
    "headache": [
        "When did the headache begin, and does it build gradually or arrive suddenly?",
        "Do you feel neck stiffness, vision changes, or confusion along with the headache?",
        "What helps or worsens the head pain (rest, light, noise, medication)?",
    ],
    "cough": [
        "Is the cough dry or producing mucus or blood?",
        "Does anything make the cough better or worse, such as lying flat?",
        "Have you experienced wheezing or fever with the cough?",
    ],
    "shortness of breath": [
        "When did the breathing difficulty start, and is it constant or episodic?",
        "Do you feel short of breath at rest, on exertion, or when lying flat?",
        "Do you use inhalers or oxygen at home, and have they helped today?",
    ],
    "chest pain": [
        "Can you describe the chest pain (pressure, sharp, tightness) and where it spreads?",
        "Did the pain start suddenly, and is it linked to activity or breathing?",
        "Do you have nausea, sweating, dizziness, or jaw/arm pain along with it?",
    ],
}

GENERAL_FOLLOW_UP_QUESTIONS = [
    "Do you have chest pain, shortness of breath, or difficulty swallowing?",
    "What medications or self-care steps have you already tried today?",
    "Have your symptoms changed in severity or pattern compared with yesterday?",
]

HISTORY_RISK_FACTORS = [
    {
        "keywords": {"diabetes", "type 1", "type 2"},
        "symptoms": {"fever", "wound", "infection"},
        "note": "Diabetes can lower immune response—new fevers or infections should be triaged promptly.",
    },
    {
        "keywords": {"asthma", "copd", "chronic lung", "emphysema"},
        "symptoms": {"shortness of breath", "cough"},
        "note": "Chronic lung conditions increase risk of severe breathing trouble from new symptoms.",
    },
    {
        "keywords": {"heart", "cardiac", "hypertension", "blood pressure", "arrhythmia"},
        "symptoms": {"chest pain", "shortness of breath"},
        "note": "Cardiac history raises worry that chest pain or breathlessness is heart-related—escalate quickly.",
    },
    {
        "keywords": {"pregnant", "pregnancy"},
        "symptoms": {"fever", "abdominal pain", "shortness of breath"},
        "note": "During pregnancy these symptoms can threaten parent or baby, so clinician review is urgent.",
    },
]

SYMPTOM_LEXICON = {
    "fever": {
        "aliases": ["fever", "temperature", "temp"],
        "red_flag_terms": ["high fever", "very high fever", "fever won't go down"],
        "fever_threshold": 102,
    },
    "headache": {
        "aliases": ["headache", "head pain", "migraine", "pounding head"],
        "red_flag_terms": [
            "worst headache",
            "stiff neck",
            "blurred vision",
            "confusion",
            "vision changes",
            "double vision",
        ],
    },
    "sore throat": {
        "aliases": ["sore throat", "throat pain", "throat hurts", "scratchy throat", "difficulty swallowing"],
        "red_flag_terms": ["difficulty swallowing", "can't swallow", "drooling", "stridor"],
    },
    "cough": {
        "aliases": ["cough", "coughing"],
        "red_flag_terms": ["blood", "bloody cough", "whooping"],
    },
    "shortness of breath": {
        "aliases": ["shortness of breath", "can't breathe", "trouble breathing", "difficult to breathe"],
        "red_flag_terms": ["rest", "lying down"],
    },
    "chest pain": {
        "aliases": ["chest pain", "tightness", "pressure"],
        "red_flag_terms": ["radiating", "left arm", "jaw pain"],
    },
}


@dataclass
class PatientProfile:
    age: Optional[str] = None
    gender: Optional[str] = None
    history: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    name: Optional[str] = None

    def render(self) -> str:
        lines: List[str] = []
        if self.name:
            lines.append(f"Name: {self.name}")
        if self.age:
            lines.append(f"Age: {self.age}")
        if self.gender:
            lines.append(f"Gender: {self.gender}")
        if self.history:
            lines.append("Relevant history: " + "; ".join(self.history))
        if self.medications:
            lines.append("Medications: " + "; ".join(self.medications))
        if self.allergies:
            lines.append("Allergies: " + "; ".join(self.allergies))
        return "\n".join(lines) or "No structured history provided."


@dataclass
class ConversationTurn:
    speaker: str
    text: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def serialise(self) -> Dict[str, str]:
        return {
            "speaker": self.speaker,
            "timestamp": self.timestamp.isoformat() + "Z",
            "text": self.text,
        }


class ConversationState:
    def __init__(self) -> None:
        self.turns: List[ConversationTurn] = []
        self.known_symptoms: List[str] = []
        self.known_red_flags: List[str] = []
        self.emotional_markers: List[str] = []
        self.symptom_records: Dict[str, Dict[str, object]] = {}
        self.latest_preprocessed: Dict[str, object] = {"cleaned": "", "tokens": []}

    def add_turn(self, speaker: str, text: str) -> Optional[Dict[str, object]]:
        self.turns.append(ConversationTurn(speaker=speaker, text=text))
        if speaker == "patient":
            return self._process_patient_text(text)
        return None

    def _process_patient_text(self, text: str) -> Dict[str, object]:
        preprocessed = normalise_patient_text(text)
        self.latest_preprocessed = preprocessed
        symptom_signals = extract_symptom_signals(text, preprocessed.get("tokens"))
        detected_symptoms = list(symptom_signals.keys())
        new_symptoms: List[str] = []
        new_red_flags: List[str] = []
        timestamp = datetime.now(UTC).isoformat()

        for symptom, details in symptom_signals.items():
            existing_record = self.symptom_records.get(symptom, {})
            prior_severity = existing_record.get("severity", "baseline")

            record = {
                **existing_record,
                **details,
                "last_reported": timestamp,
            }
            if "first_reported" not in record:
                record["first_reported"] = timestamp

            self.symptom_records[symptom] = record

            if symptom not in self.known_symptoms:
                new_symptoms.append(symptom)
            if details.get("severity") == "red_flag" and prior_severity != "red_flag":
                triggers = details.get("triggers")
                if isinstance(triggers, list) and triggers:
                    new_red_flags.extend(triggers)
                else:
                    new_red_flags.append(symptom)

        self.known_symptoms = sorted(self.symptom_records.keys())

        lowered = text.lower()
        for keyword in EMERGENCY_KEYWORDS:
            if keyword in lowered and keyword not in self.known_red_flags:
                self.known_red_flags.append(keyword)
                new_red_flags.append(keyword)

        self._update_emotional_state(text)

        return {
            "detected_symptoms": detected_symptoms,
            "new_symptoms": new_symptoms,
            "symptom_signals": symptom_signals,
            "new_red_flags": list(dict.fromkeys(new_red_flags)),
            "normalized_text": preprocessed.get("cleaned", ""),
            "normalized_tokens": preprocessed.get("tokens", []),
        }

    def _update_emotional_state(self, text: str) -> None:
        lowered = text.lower()
        matches = [kw for kw in EMOTIONAL_DISTRESS_KEYWORDS if kw in lowered]
        for match in matches:
            if match not in self.emotional_markers:
                self.emotional_markers.append(match)

    def render_history(self, limit: int = 6) -> str:
        recent = self.turns[-limit:]
        formatted: List[str] = []
        for turn in recent:
            speaker_label = "Patient" if turn.speaker == "patient" else "Assistant"
            formatted.append(f"{speaker_label}: {turn.text}")
        return "\n".join(formatted) or "No prior conversation."

    def to_json(self) -> List[Dict[str, str]]:
        return [turn.serialise() for turn in self.turns]

    def render_emotional_state(self) -> str:
        if not self.emotional_markers:
            return "No emotional distress cues captured."
        return "\n".join(f"- {marker}" for marker in self.emotional_markers)


def detect_tone(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in STRESS_KEYWORDS):
        return "high_empathy"
    return "neutral"


def detect_detail_preference(text: str) -> str:
    lowered = text.lower()
    if any(marker in lowered for marker in SIMPLE_LANGUAGE_MARKERS):
        return "simple"
    if any(marker in lowered for marker in CLINICIAN_MARKERS):
        return "clinical"
    return "balanced"


def detect_urgency(
    state: ConversationState,
    latest_patient_text: str,
    new_red_flags: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str]]:
    if new_red_flags:
        cleaned = [flag for flag in new_red_flags if flag]
        if cleaned:
            return True, cleaned[-1]
    lowered = latest_patient_text.lower()
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in lowered:
            return True, keyword
    if state.known_red_flags:
        return True, state.known_red_flags[-1]
    return False, None


def detect_emotional_distress(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in EMOTIONAL_DISTRESS_KEYWORDS)


def extract_symptom_signals(raw_text: str, tokens: Optional[List[str]] = None) -> Dict[str, Dict[str, object]]:
    lowered = raw_text.lower()
    lowered_no_space = lowered.replace(" ", "")
    token_set = set(tokens or [])
    results: Dict[str, Dict[str, object]] = {}

    temp_match = re.search(r"(\d{2,3})\s*(?:°?\s*f|fahrenheit|deg f)", lowered)
    temperature_f: Optional[int] = None
    if temp_match:
        try:
            temperature_f = int(temp_match.group(1))
        except ValueError:
            temperature_f = None

    for symptom, config in SYMPTOM_LEXICON.items():
        aliases = [alias.lower() for alias in config.get("aliases", [])]
        alias_match: Optional[str] = None
        for alias in aliases:
            compact = alias.replace(" ", "")
            alias_tokens = [part for part in alias.split() if part and part not in STOPWORDS]
            if alias in lowered:
                alias_match = alias
                break
            if compact and compact in lowered_no_space:
                alias_match = alias
                break
            if alias_tokens and all(part in token_set for part in alias_tokens):
                alias_match = alias
                break
        if alias_match is None:
            continue

        severity = "baseline"
        triggers: List[str] = []

        for term in config.get("red_flag_terms", []):
            if term in lowered:
                severity = "red_flag"
                triggers.append(term)

        threshold = config.get("fever_threshold")
        if threshold and temperature_f is not None and temperature_f >= threshold:
            severity = "red_flag"
            triggers.append(f"temperature_{temperature_f}F")

        result: Dict[str, object] = {
            "raw_match": alias_match,
            "severity": severity,
            "triggers": triggers,
        }
        if threshold and temperature_f is not None:
            result["temperature_f"] = temperature_f

        results[symptom] = result

    return results


def classify_intent(
    state: ConversationState,
    text: str,
    context: Dict[str, object],
    emotional_distress: bool,
    urgent: bool,
) -> str:
    lowered = text.lower()
    has_greeting = any(lowered.startswith(greet) or f" {greet}" in lowered for greet in GREETINGS)
    new_symptoms = context.get("new_symptoms", []) or []
    detected_symptoms = context.get("detected_symptoms", []) or []

    patient_turns = [turn for turn in state.turns if turn.speaker == "patient"]
    is_first_patient_turn = len(patient_turns) == 1

    if urgent:
        return "urgent_symptom"
    if new_symptoms:
        return "symptom_report" if is_first_patient_turn else "symptom_update"
    if emotional_distress and detected_symptoms and state.symptom_records:
        return "mixed"
    if emotional_distress and not detected_symptoms:
        return "emotional_support"
    if any(keyword in lowered for keyword in FOLLOW_UP_KEYWORDS) and state.symptom_records:
        return "follow_up"
    if detected_symptoms and not new_symptoms:
        return "symptom_reinforce"
    if has_greeting and not detected_symptoms:
        return "greeting"
    if state.known_symptoms and not detected_symptoms:
        return "follow_up"
    if detected_symptoms:
        return "symptom_update"
    return "general"


def detail_instruction(detail_level: str) -> str:
    mapping = {
        "simple": "Use clear, plain language and short sentences.",
        "clinical": "Provide more clinical depth while remaining understandable.",
        "balanced": "Give a balanced mix of accessible language with moderate detail.",
    }
    return mapping.get(detail_level, "Provide balanced, patient-friendly explanations.")


def format_new_information(new_symptoms: List[str], new_red_flags: List[str]) -> str:
    sections: List[str] = []
    if new_symptoms:
        sections.append("New symptoms noted: " + ", ".join(new_symptoms))
    if new_red_flags:
        sections.append("New red-flag cues: " + ", ".join(new_red_flags))
    return "\n".join(sections) or "No new symptom information detected in the latest message."


def format_reasoning_notes(reasoning: List[str]) -> str:
    if not reasoning:
        return "No specific clinical reasoning cues surfaced."
    limited = reasoning[:3]
    formatted = "\n".join(f"- {line}" for line in limited)
    if len(reasoning) > 3:
        formatted += "\n- Additional considerations recorded in clinician view."
    return formatted


def format_follow_up_questions(questions: List[str]) -> str:
    if not questions:
        return "No additional follow-up questions required at this time."
    return "\n".join(f"- {question}" for question in questions)


def format_red_flag_actions(actions: List[str]) -> str:
    if not actions:
        return "No immediate emergency actions beyond standard monitoring."
    return "\n".join(f"- {action}" for action in actions)


def _natural_join(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def generate_patient_reflection(
    state: ConversationState,
    profile: PatientProfile,
    new_symptoms: List[str],
) -> str:
    if new_symptoms:
        prior = [symptom for symptom in state.known_symptoms if symptom not in new_symptoms]
        new_phrase = _natural_join(new_symptoms)
        if prior:
            prior_phrase = _natural_join(prior)
            return f"I hear you’re still dealing with {prior_phrase}, and now you’re also noticing {new_phrase}."
        return f"I hear you’re experiencing {new_phrase}."
    if state.known_symptoms:
        current = _natural_join(state.known_symptoms)
        return f"I’m keeping track of the {current} you’ve mentioned—how have those changed since we last spoke?"
    if profile.history:
        return "Thanks for checking in—let’s capture what’s happening so I can tailor guidance to your history."
    return "I’m here with you—tell me more so we can work through this together."


def generate_empathy_line(
    emotional_markers: List[str],
    tone: str,
    primary_symptom: Optional[str],
) -> str:
    if emotional_markers:
        feeling = emotional_markers[-1]
        return f"It’s understandable to feel {feeling}—we’ll work through this carefully."
    if tone == "high_empathy" and primary_symptom:
        return f"A {primary_symptom} can feel draining—let’s unpack what might be driving it."
    if tone == "high_empathy":
        return "I know this can feel worrying—let’s take it step by step."
    if primary_symptom:
        return f"Let’s look closely at this {primary_symptom} and make sure you’re safe."
    return "I’ll keep the guidance clear and practical while we sort this out."


def decide_response_length(intent: str, urgent: bool, new_symptoms: List[str], default: str = "detailed") -> str:
    if urgent or intent == "urgent_symptom":
        return "detailed"
    if new_symptoms and intent not in {"follow_up", "symptom_reinforce"}:
        return "detailed"
    if intent in {"follow_up", "symptom_reinforce"}:
        return "brief"
    return default


def generate_progressive_summary(
    intent: str,
    urgent: bool,
    new_symptoms: List[str],
    known_symptoms: List[str],
    trigger: Optional[str],
) -> str:
    if urgent:
        return "This raises emergency concerns—let’s get you connected to in-person care right away."
    if intent == "greeting":
        return "Let’s start by capturing exactly what you’re feeling so I can advise clearly."
    if new_symptoms:
        joined = _natural_join(new_symptoms)
        if known_symptoms:
            previous = _natural_join([s for s in known_symptoms if s not in new_symptoms])
            if previous:
                return f"Noted {joined} on top of the {previous} you’d already shared."
        return f"I’m noting {joined} so we can tailor next steps."
    if intent in {"follow_up", "symptom_reinforce"}:
        return "Let’s check how your earlier symptoms are evolving and keep the safety plan current."
    if intent == "emotional_support":
        return "We’ll steady the emotional side while keeping an eye on any danger signs."
    if trigger:
        return f"Keeping watch for changes related to {trigger}."
    if known_symptoms:
        return "Continuing to monitor " + _natural_join(known_symptoms)
    return "Share any symptoms you’re noticing so we can map out next steps."


def _history_matches(profile: PatientProfile) -> Tuple[str, List[str]]:
    history_strings = [entry.lower() for entry in profile.history]
    history_blob = " ".join(history_strings)
    return history_blob, history_strings


def generate_clinical_reasoning(
    state: ConversationState,
    profile: PatientProfile,
    new_red_flags: List[str],
    focus_symptoms: List[str],
) -> List[str]:
    if not state.symptom_records:
        return []

    history_blob, _ = _history_matches(profile)
    reasoning: List[str] = []
    focus_lower = {entry.lower() for entry in (focus_symptoms or state.known_symptoms or [])}
    if not focus_lower:
        focus_lower = {entry.lower() for entry in state.symptom_records.keys()}

    for symptom, record in state.symptom_records.items():
        symptom_lower = symptom.lower()
        if focus_lower and symptom_lower not in focus_lower:
            continue
        parts: List[str] = []
        severity = str(record.get("severity", "baseline"))
        triggers = record.get("triggers") or []
        temperature = record.get("temperature_f")

        if severity == "red_flag":
            trigger_text = f" ({', '.join(triggers)})" if triggers else ""
            parts.append(f"Red-flag pattern detected{trigger_text}—treat {symptom_lower} as high risk.")

        if temperature:
            parts.append(f"Recorded temperature {temperature}°F increases concern for infection complications.")

        for risk in HISTORY_RISK_FACTORS:
            if symptom_lower not in risk["symptoms"]:
                continue
            if any(keyword in history_blob for keyword in risk["keywords"]):
                parts.append(risk["note"])

        if not parts:
            parts.append(f"We’ll track how {symptom_lower} evolves to separate routine illness from anything serious.")

        if parts:
            reasoning.append(f"For {symptom_lower}: {' '.join(parts)}")

    for flag in new_red_flags:
        flag_lower = flag.lower()
        for keyword, (category, advice) in RED_FLAG_CATEGORY_MAP.items():
            if keyword in flag_lower:
                reasoning.append(f"{category}: {advice}")
                break

    if profile.history and not reasoning:
        reasoning.append("Patient history noted but no symptom-specific risks detected yet.")

    return reasoning


def generate_follow_up_questions(state: ConversationState, new_symptoms: List[str]) -> List[str]:
    questions: List[str] = []
    targets = new_symptoms or state.known_symptoms
    seen = set()

    for symptom in targets:
        follow_ups = SYMPTOM_FOLLOW_UP_PROMPTS.get(symptom.lower())
        if not follow_ups:
            continue
        for prompt in follow_ups:
            if prompt not in seen:
                questions.append(prompt)
                seen.add(prompt)

    for default_question in GENERAL_FOLLOW_UP_QUESTIONS:
        if default_question not in seen:
            questions.append(default_question)
            seen.add(default_question)

    if not questions:
        return [
            "Could you share when each symptom started, how severe it feels now, and what makes it better or worse?",
            GENERAL_FOLLOW_UP_QUESTIONS[0],
        ]
    return questions[:6]


def summarise_red_flag_actions(red_flags: List[str]) -> List[str]:
    actions: List[str] = []
    seen = set()
    for flag in red_flags:
        lowered = flag.lower()
        if lowered.startswith("temperature_") and "Systemic infection concern" not in seen:
            actions.append(
                "Systemic infection concern: Persistent fevers above 102°F should be reviewed urgently by a clinician."
            )
            seen.add("Systemic infection concern")
        for keyword, (category, advice) in RED_FLAG_CATEGORY_MAP.items():
            if keyword in lowered and category not in seen:
                actions.append(f"{category}: {advice}")
                seen.add(category)
    return actions


def evaluate_response_metadata(candidate: Dict[str, object]) -> Tuple[Dict[str, float], List[str]]:
    """Generate lightweight heuristic scores used for active learning triage."""

    scores: Dict[str, float] = {
        "safety": 0.7,
        "empathy": 0.6,
        "doctor_like": 0.6,
    }
    review_reasons: List[str] = []

    red_flag_actions = candidate.get("red_flag_actions") or []
    urgent = bool(candidate.get("urgent"))
    tone = candidate.get("tone")
    clinical_reasoning = candidate.get("clinical_reasoning") or []

    if urgent and red_flag_actions:
        scores["safety"] = 0.95
    elif urgent and not red_flag_actions:
        scores["safety"] = 0.2
        review_reasons.append("urgent_without_escalation")
    else:
        scores["safety"] = 0.75 if red_flag_actions else 0.65

    if tone == "high_empathy" or candidate.get("empathy_line"):
        scores["empathy"] = 0.9
    if candidate.get("emotional_distress") and tone != "high_empathy":
        scores["empathy"] = min(scores["empathy"], 0.5)
        review_reasons.append("distress_without_high_empathy")

    if clinical_reasoning:
        scores["doctor_like"] = 0.85
    if candidate.get("response_focus") in {"general", "greeting"} and not clinical_reasoning:
        scores["doctor_like"] = 0.55

    if scores["safety"] < 0.5 or scores["empathy"] < 0.5:
        review_reasons.append("low_auto_score")

    return scores, review_reasons


STRUCTURED_PROMPT_TEMPLATE = """
You are a clinical support assistant helping a remote triage nurse interpret patient updates. Use the structured context below to craft a safe, empathetic response.

[Patient Context]
{patient_context}

[Conversation Summary]
{conversation_summary}

[Current Symptoms]
{symptom_summary}

[Emotional State]
{emotional_summary}

[New Information Detected]
{new_information}

[Red Flags Detected]
{red_flags}

[Patient Reflection Anchor]
{patient_reflection}

[Empathy Anchor]
{empathy_line}

[Clinical Reasoning Cues]
{reasoning_notes}

[Follow-Up Focus Areas]
{follow_up_focus}

[Emergency Actions]
{red_flag_actions}

[Progressive Summary Cue]
{progressive_summary}

[Response Instructions]
- Begin with the Patient Reflection Anchor in natural language before adding new information.
- Let the Empathy Anchor influence word choice and pacing.
- Calibrate overall length: {response_length_instruction}
- Provide an **Explanation**: {explanation_instruction}
- Offer **Empathetic communication**: {empathy_instruction}
- Ask 2-3 **Clarifying questions** using the Follow-Up Focus Areas.
- Outline **Safe next steps**: {safety_instruction}
- Provide **Supportive guidance**: {support_instruction}
- Include **Disclaimer**: remind user this is educational and not a substitute for emergency or in-person care.
- Apply language style: {detail_instruction}
- Verify adherence to institutional safety policies.

Self-check before final answer:
1. Did you consider all red flags and escalate appropriately?
2. Is the advice clinically safe and non-prescriptive?
3. Are medication or diagnostic suggestions accompanied by disclaimers to contact a professional?
4. If any emergency concern exists, begin the answer with: "⚠️ This could be a serious condition..." and direct immediate care.

Respond in markdown with headings: "Summary", "Explanation", "Empathetic communication", "Clarifying questions", "Safe next steps", "Disclaimer".
"""


class AdaptivePromptEngine:
    def __init__(self, patient_profile: Optional[PatientProfile] = None, state: Optional[ConversationState] = None):
        self.patient_profile = patient_profile or PatientProfile()
        self.state = state or ConversationState()

    def ingest_patient_message(self, text: str) -> Dict[str, object]:
        context = self.state.add_turn("patient", text) or {
            "detected_symptoms": [],
            "new_symptoms": [],
            "symptom_signals": {},
            "new_red_flags": [],
        }
        tone = detect_tone(text)
        detail_pref = detect_detail_preference(text)
        emotional_distress = detect_emotional_distress(text)
        urgent, trigger = detect_urgency(self.state, text, context.get("new_red_flags") or [])
        return {
            "tone": tone,
            "detail": detail_pref,
            "urgent": urgent,
            "trigger": trigger,
            "emotional_distress": emotional_distress,
            "context": context,
        }

    def ingest_assistant_message(self, text: str) -> None:
        self.state.add_turn("assistant", text)

    def _summarise_symptoms(self) -> str:
        if not self.state.known_symptoms:
            return "No explicit symptoms captured yet."
        return "\n".join(f"- {symptom}" for symptom in self.state.known_symptoms)

    def _summarise_red_flags(self) -> str:
        if not self.state.known_red_flags:
            return "None detected. Continue screening."
        return "\n".join(f"- {flag}" for flag in self.state.known_red_flags)

    def _summarise_emotional(self) -> str:
        return self.state.render_emotional_state()

    def build_prompt(self, patient_latest: str) -> Dict[str, object]:
        adaptation = self.ingest_patient_message(patient_latest)
        context = adaptation.get("context", {}) or {}
        new_symptoms: List[str] = list(context.get("new_symptoms", []))  # type: ignore[arg-type]
        detected_symptoms: List[str] = list(context.get("detected_symptoms", []))  # type: ignore[arg-type]
        new_red_flags: List[str] = list(context.get("new_red_flags", []))  # type: ignore[arg-type]
        symptom_signals = context.get("symptom_signals", {})

        empathy_instruction = (
            "Use high empathy, validate feelings, slow the pace, and reassure while staying realistic."
            if adaptation["tone"] == "high_empathy"
            else "Offer compassionate and professional reassurance."
        )

        detail_preference = adaptation["detail"]

        emotional_summary = self._summarise_emotional()
        new_information = format_new_information(new_symptoms, new_red_flags)

        focus_symptoms = new_symptoms or self.state.known_symptoms
        reasoning_notes = generate_clinical_reasoning(
            self.state,
            self.patient_profile,
            new_red_flags,
            focus_symptoms,
        )
        patient_reflection = generate_patient_reflection(self.state, self.patient_profile, new_symptoms)
        primary_symptom = focus_symptoms[0] if focus_symptoms else None
        empathy_line = generate_empathy_line(self.state.emotional_markers, adaptation["tone"], primary_symptom)
        follow_up_questions = generate_follow_up_questions(self.state, new_symptoms)
        red_flag_actions = summarise_red_flag_actions(self.state.known_red_flags or new_red_flags)
        if adaptation["urgent"] and not red_flag_actions:
            red_flag_actions = [
                "Emergency suspicion raised—direct the patient to call local emergency services immediately and stay with them on the line.",
            ]

        intent = classify_intent(
            self.state,
            patient_latest,
            context,
            adaptation["emotional_distress"],
            adaptation["urgent"],
        )

        category: ResponseCategory = get_category(intent, adaptation["urgent"])

        if detail_preference == "balanced":
            detail_preference = category.detail_bias
        detail_instr = detail_instruction(detail_preference)

        if category.default_follow_ups:
            follow_up_questions = _dedupe_preserve_order(list(category.default_follow_ups) + follow_up_questions)[:6]

        progressive_summary = generate_progressive_summary(
            intent,
            adaptation["urgent"],
            new_symptoms,
            self.state.known_symptoms,
            adaptation.get("trigger"),
        )

        response_length = decide_response_length(intent, adaptation["urgent"], new_symptoms, category.response_length)
        if response_length == "brief":
            response_length_instruction = "Keep the spoken response concise—roughly two short paragraphs—and prioritise reflection, key question, and safety reminder."
        else:
            response_length_instruction = "Offer a structured response with clear headings so the clinician can review details."

        explanation_instruction = category.explanation_instruction
        safety_instruction = category.safety_instruction
        support_instruction = category.support_instruction
        response_focus = category.response_focus
        resource_type = category.resource_type

        conversation_summary = self.state.render_history()
        symptom_summary = self._summarise_symptoms()
        red_flags = self._summarise_red_flags()

        prompt = STRUCTURED_PROMPT_TEMPLATE.format(
            patient_context=self.patient_profile.render(),
            conversation_summary=conversation_summary,
            symptom_summary=symptom_summary,
            emotional_summary=emotional_summary,
            new_information=new_information,
            red_flags=red_flags,
            patient_reflection=patient_reflection,
            empathy_line=empathy_line,
            reasoning_notes=format_reasoning_notes(reasoning_notes),
            follow_up_focus=format_follow_up_questions(follow_up_questions),
            red_flag_actions=format_red_flag_actions(red_flag_actions),
            progressive_summary=progressive_summary,
            response_length_instruction=response_length_instruction,
            explanation_instruction=explanation_instruction,
            empathy_instruction=empathy_instruction,
            safety_instruction=safety_instruction,
            support_instruction=support_instruction,
            detail_instruction=detail_instr,
        )

        if adaptation["urgent"] or intent == "urgent_symptom":
            prompt += (
                "\nMandatory emergency directive: If the emergency keyword appears, lead with a clear warning and advise immediate emergency services."
            )

        metadata: Dict[str, object] = {
            "tone": adaptation["tone"],
            "detail_preference": detail_preference,
            "urgent": adaptation["urgent"],
            "trigger": adaptation["trigger"],
            "emotional_distress": adaptation["emotional_distress"],
            "emotional_markers": self.state.emotional_markers,
            "category": category.name,
            "response_focus": response_focus,
            "resource_type": resource_type,
            "intent": intent,
            "new_symptoms": new_symptoms,
            "detected_symptoms": detected_symptoms,
            "known_symptoms": self.state.known_symptoms,
            "new_red_flags": new_red_flags,
            "new_information": new_information,
            "symptom_signals": symptom_signals,
            "clinical_reasoning": reasoning_notes,
            "follow_up_questions": follow_up_questions,
            "red_flag_actions": red_flag_actions,
            "progressive_summary": progressive_summary,
            "reflection": patient_reflection,
            "empathy_line": empathy_line,
            "response_length": response_length,
            "response_length_instruction": response_length_instruction,
            "normalized_tokens": context.get("normalized_tokens", []),
            "normalized_text": context.get("normalized_text"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        auto_scores, review_reasons = evaluate_response_metadata(metadata)
        metadata["auto_scores"] = auto_scores
        metadata["needs_review"] = bool(review_reasons)
        metadata["review_reasons"] = review_reasons

        return {"prompt": prompt, "metadata": metadata}

    def export_state(self, path: Path) -> None:
        payload = {
            "patient_profile": self.patient_profile.__dict__,
            "conversation_turns": self.state.to_json(),
            "known_symptoms": self.state.known_symptoms,
            "known_red_flags": self.state.known_red_flags,
            "emotional_markers": self.state.emotional_markers,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def integrate_correction_feedback(self, annotation_path: Path) -> None:
        if not annotation_path.exists():
            return
        unsafe_cases: List[Dict[str, object]] = []
        needs_follow_up: List[Dict[str, object]] = []
        with annotation_path.open("r", encoding="utf-8") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    if "unsafe" in raw:
                        unsafe_cases.append({"raw": raw})
                    continue
                resolution = str(entry.get("resolution", "")).lower()
                if resolution == "unsafe":
                    unsafe_cases.append(entry)
                elif resolution == "needs_follow_up":
                    needs_follow_up.append(entry)

        if unsafe_cases or needs_follow_up:
            print(
                "Loaded feedback entries needing action: "
                f"{len(unsafe_cases)} unsafe, {len(needs_follow_up)} follow-up. "
                "Consider updating prompts, guardrails, or escalation policies accordingly."
            )
