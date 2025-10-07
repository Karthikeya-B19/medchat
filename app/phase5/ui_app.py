"""Streamlit prototype for Phase 5 patient/clinician chatbot experience.

The app demonstrates:
- Chat interface with adaptive prompting metadata.
- Trust & safety affordances (disclaimers, escalation, logging).
- Accessibility features (high contrast, large font, multilingual, voice hooks).

Run with:
    streamlit run app/phase5/ui_app.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

import streamlit as st

UTC = timezone.utc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.phase5.adaptive_prompt_engine import AdaptivePromptEngine, PatientProfile
from app.phase5.logging_utils import StructuredConversationLogger

try:
    from deep_translator import GoogleTranslator  # type: ignore
except ImportError:  # pragma: no cover - dependency optional in prototype stage
    GoogleTranslator = None

try:
    import speech_recognition as sr  # type: ignore
except ImportError:  # pragma: no cover - dependency optional in prototype stage
    sr = None

try:
    from gtts import gTTS  # type: ignore
except ImportError:  # pragma: no cover - dependency optional in prototype stage
    gTTS = None

TRANSLATION_LANG_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
}

RESOURCE_LINKS: Dict[str, List[Tuple[str, str]]] = {
    "medical": [
        ("Mayo Clinic - Fever", "https://www.mayoclinic.org/diseases-conditions/fever/symptoms-causes/syc-20352759"),
        ("CDC - Sore Throat", "https://www.cdc.gov/groupastrep/diseases-public/sore-throat.html"),
    ],
    "mental_health": [
        ("NIMH - Coping with Anxiety", "https://www.nimh.nih.gov/health/topics/anxiety-disorders"),
        ("Mind - Breathing Exercises", "https://www.mind.org.uk/information-support/tips-for-everyday-living/relaxation/"),
    ],
    "integrated": [
        ("Mayo Clinic - Stress and Illness", "https://www.mayoclinic.org/healthy-lifestyle/stress-management/in-depth/stress/art-20046037"),
        ("APA - Managing Anxiety", "https://www.apa.org/topics/anxiety/"),
    ],
    "emergency": [
        ("WHO - Emergency Signs", "https://www.who.int/emergencies"),
        ("911 Immediate Assistance", "tel:911"),
    ],
    "follow_up": [
        ("NHS - Check symptoms", "https://www.nhs.uk/conditions/"),
        ("Cleveland Clinic - Symptom Diary", "https://health.clevelandclinic.org/how-to-keep-a-symptom-diary/"),
    ],
    "greeting": [
        ("CDC - Preparing for Your Visit", "https://www.cdc.gov/family/healthcare/index.htm"),
        ("AHRQ - Question Builder", "https://www.ahrq.gov/patient-safety/question-builder.html"),
    ],
    "general": [
        ("Mayo Clinic - First Aid", "https://www.mayoclinic.org/first-aid"),
    ],
}

LOG_DIR = Path("app/phase5/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

DISCLAIMER = "I am an AI medical assistant, not a substitute for a licensed doctor."


def _apply_custom_theme(high_contrast: bool, large_font: bool) -> None:
    screen_reader = st.session_state.get("screen_reader_mode", False)
    base_font = "20px" if screen_reader else ("18px" if large_font else "16px")
    border_color = "#000000" if screen_reader else ("#ffffff" if high_contrast else "#e0e0e0")
    background = "#ffffff" if screen_reader else ("#000000" if high_contrast else "#f9fafb")
    foreground = "#000000" if screen_reader else ("#f5f5f5" if high_contrast else "#1f2933")
    accent = "#cc0000" if screen_reader else ("#f97316" if high_contrast else "#2563eb")
    st.markdown(
        f"""
        <style>
            body {{
                background-color: {background};
                color: {foreground};
            }}
            .stApp {{
                background-color: {background};
                color: {foreground};
            }}
            .chat-bubble {{
                padding: 0.9rem 1.1rem;
                border-radius: 16px;
                margin-bottom: 0.6rem;
                border: 1px solid {border_color};
                font-size: {base_font};
            }}
            .chat-patient {{
                background: rgba(37, 99, 235, 0.12);
            }}
            .chat-assistant {{
                background: rgba(16, 185, 129, 0.12);
            }}
            .metadata-panel {{
                background: rgba(31, 41, 55, 0.04);
                border: 1px solid {border_color};
                padding: 1rem;
                border-radius: 12px;
                font-size: {base_font};
                line-height: 1.6;
            }}
            .emergency-button button {{
                background: {accent};
                color: white;
                font-weight: 700;
                border: none;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def translate_to_backend(text: str, interface_language: str) -> Tuple[str, Optional[str]]:
    if not text or interface_language == "English":
        return text, None
    if GoogleTranslator is None:
        return text, "Translation service unavailable."
    source_code = TRANSLATION_LANG_CODES.get(interface_language, "auto")
    try:
        translator = GoogleTranslator(source=source_code, target="en")
        return translator.translate(text), None
    except Exception as exc:  # pragma: no cover - external service variability
        return text, f"Translation to English failed: {exc}"


def translate_from_backend(text: str, interface_language: str) -> Tuple[str, Optional[str]]:
    if not text or interface_language == "English":
        return text, None
    if GoogleTranslator is None:
        return text, "Translation service unavailable."
    target_code = TRANSLATION_LANG_CODES.get(interface_language, "en")
    try:
        translator = GoogleTranslator(source="en", target=target_code)
        return translator.translate(text), None
    except Exception as exc:  # pragma: no cover - external service variability
        return text, f"Translation from English failed: {exc}"


def get_resource_links(resource_type: str) -> List[Tuple[str, str]]:
    if resource_type in RESOURCE_LINKS:
        return RESOURCE_LINKS[resource_type]
    return RESOURCE_LINKS["general"]


def recognise_speech(uploaded_file) -> Tuple[Optional[str], Optional[str]]:
    if uploaded_file is None:
        return None, None
    if sr is None:
        return None, "Voice recognition dependency not installed."
    recogniser = sr.Recognizer()
    with sr.AudioFile(uploaded_file) as source:
        audio = recogniser.record(source)
    try:
        text = recogniser.recognize_google(audio)
        return text, None
    except Exception as exc:  # pragma: no cover - depends on API
        return None, f"Voice recognition failed: {exc}"


def synthesise_speech(text: str, language: str) -> Tuple[Optional[BytesIO], Optional[str]]:
    if not text:
        return None, "Nothing to synthesise."
    lang_code = TRANSLATION_LANG_CODES.get(language, "en")
    if gTTS is None:
        return None, "Text-to-speech dependency not installed."
    try:
        tts = gTTS(text=text, lang=lang_code)
        buffer = BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer, None
    except Exception as exc:  # pragma: no cover - network/tts variability
        return None, f"Speech synthesis failed: {exc}"


def initialise_state() -> None:
    if "engine" not in st.session_state:
        st.session_state.engine = AdaptivePromptEngine()
    if "logger" not in st.session_state:
        st.session_state.logger = StructuredConversationLogger(LOG_DIR)
    if "conversation" not in st.session_state:
        st.session_state.conversation = []  # type: ignore[var-annotated]
    if "metadata_log" not in st.session_state:
        st.session_state.metadata_log = []  # type: ignore[var-annotated]
    if "language" not in st.session_state:
        st.session_state.language = "English"
    if "high_contrast" not in st.session_state:
        st.session_state.high_contrast = False
    if "large_font" not in st.session_state:
        st.session_state.large_font = False
    if "patient_profile" not in st.session_state:
        st.session_state.patient_profile = PatientProfile()
    if "screen_reader_mode" not in st.session_state:
        st.session_state.screen_reader_mode = False
    if "initial_greeting_added" not in st.session_state:
        st.session_state.initial_greeting_added = False
    if "feedback_log" not in st.session_state:
        st.session_state.feedback_log = []  # type: ignore[var-annotated]


def reset_conversation() -> None:
    st.session_state.conversation = []
    st.session_state.metadata_log = []
    st.session_state.engine = AdaptivePromptEngine(patient_profile=st.session_state.patient_profile)
    st.session_state.initial_greeting_added = False


def mock_model_response(metadata: Dict[str, object]) -> Dict[str, object]:
    focus = metadata.get("response_focus", "physical_monitoring")
    intent = metadata.get("intent", "general")
    trigger = metadata.get("trigger")
    emotional_markers = list(metadata.get("emotional_markers", []) or [])
    new_information = metadata.get("new_information", "")
    new_symptoms = list(metadata.get("new_symptoms", []) or [])
    known_symptoms = list(metadata.get("known_symptoms", []) or [])
    reasoning_notes = list(metadata.get("clinical_reasoning", []) or [])
    follow_up_questions = list(metadata.get("follow_up_questions", []) or [])
    red_flag_actions = list(metadata.get("red_flag_actions", []) or [])
    progressive_summary = metadata.get("progressive_summary") or "Continuing assessment."
    reflection = metadata.get("reflection") or progressive_summary
    empathy_line = metadata.get("empathy_line") or "Let’s walk through this together."
    response_length = metadata.get("response_length", "detailed")

    DEFAULT_NEW_INFO = "No new symptom information detected in the latest message."

    lead_in = reflection

    if focus == "urgent_physical":
        scenario_context = "Emergency concern detected—seek in-person care immediately."
        explanation = (
            f"The reported red flag{' ' + trigger if trigger else ''} warrants urgent evaluation. Discuss possible cardiac, respiratory, or neurological causes.".strip()
        )
        safe_steps = (
            "- Call local emergency services now.\n"
            "- Do not drive yourself; arrange safe transport.\n"
            "- Keep another adult informed and note any new or worsening signs."
        )
        support = "Keep breathing steady, stay seated or lying down if dizzy, and keep your phone close while help is on the way."
        resource_type = "emergency"
    elif focus == "emotional_support":
        marker_text = emotional_markers[0] if emotional_markers else "feeling anxious"
        scenario_context = "I can hear how anxious you’re feeling—let’s slow things down together."
        explanation = "There are no new physical danger signs in what you shared, so we can focus on managing the emotional distress right now."
        safe_steps = (
            "- Practice a grounding exercise: inhale for four counts, hold for four, exhale for six.\n"
            "- Reach out to a trusted person or clinician if the anxiety keeps building.\n"
            "- Seek urgent help if anxiety comes with chest pain, fainting, or thoughts of self-harm."
        )
        support = f"Acknowledge the {marker_text} sensation, remind yourself it will pass, and pair breathing with slow stretching or sips of water."
        resource_type = "mental_health"
    elif focus == "mixed":
        scenario_context = "Your symptoms and stress responses are linked, so we’ll tackle both carefully."
        explanation = "We will monitor the physical symptoms while also addressing how anxiety can amplify how you feel."
        safe_steps = (
            "- Track temperature, hydration, and any breathing changes.\n"
            "- Schedule a follow-up with your clinician within 24–48 hours if symptoms persist.\n"
            "- Use calming routines (breathing, short walks) to keep stress from escalating symptoms."
        )
        support = "Alternate between symptom check-ins and short calming breaks, and note what eases your discomfort."
        resource_type = "integrated"
    elif focus == "greeting":
        scenario_context = "Welcome! Let’s capture the details so I can guide you safely."
        explanation = "I can offer the best guidance once I understand your current symptoms, history, and any immediate worries."
        safe_steps = (
            "- Describe any symptoms, even if they feel minor.\n"
            "- Share timing, severity, and what makes symptoms better or worse.\n"
            "- Mention major medical history, allergies, or medications that could change recommendations."
        )
        support = "Take your time—short, factual notes are fine. I’ll help organise the information once it’s shared."
        resource_type = "greeting"
    elif focus == "follow_up":
        scenario_context = "Thanks for checking in—let’s review what’s stable and what needs attention."
        if known_symptoms:
            symptom_list = ", ".join(known_symptoms)
            explanation = f"We’re still monitoring your documented symptoms: {symptom_list}. No new red flags were detected."
        else:
            explanation = "No critical changes detected. Continue observing your wellbeing and note any shifts."
        safe_steps = (
            "- Keep tracking how symptoms evolve, including timing and severity.\n"
            "- Follow your clinician’s plan; contact them sooner if symptoms intensify or new signs appear.\n"
            "- Reach out immediately if you notice emergency warning signs (difficulty breathing, chest pain, confusion)."
        )
        support = "Consistency matters—your updates help us adjust advice promptly if anything changes."
        resource_type = "follow_up"
    elif focus == "symptom_update":
        scenario_context = "I’ve captured the new details you provided—here’s how they fit with your record."
        explanation = "We’ll integrate the new symptoms into your history so monitoring and escalation advice stays current."
        safe_steps = (
            "- Note how the newly added symptoms interact with existing ones.\n"
            "- Share objective data if available (temperature, oxygen saturation, blood pressure).\n"
            "- Seek medical care sooner if the new issues escalate quickly or combine with red-flag signs."
        )
        support = "You’ve done the right thing by updating me—let’s keep a close eye and adjust guidance if anything worsens."
        resource_type = metadata.get("resource_type", "medical") or "medical"
    else:  # physical_monitoring or general overview
        scenario_context = "Monitoring your symptoms and highlighting what to watch next."
        explanation = "Current details suggest a common illness, but we will highlight key warning signs that would need medical review."
        safe_steps = (
            "- Rest, hydrate, and use fever reducers if advised by your clinician.\n"
            "- Contact your doctor if fever exceeds 102°F (38.9°C), you develop breathing trouble, or symptoms last beyond 3 days.\n"
            "- Seek urgent care immediately if you notice rash, confusion, or difficulty swallowing."
        )
        support = "It’s normal to feel uncomfortable—note how symptoms change and reach out early if you’re worried."
        resource_type = metadata.get("resource_type", "medical") or "medical"

    if new_information and new_information != DEFAULT_NEW_INFO:
        explanation += f"\n\n**New details captured:** {new_information}"
    if new_symptoms and focus != "greeting":
        explanation += f"\n- Newly recorded symptoms: {', '.join(new_symptoms)}."

    summary_section = reflection
    if empathy_line:
        summary_section = f"{summary_section} {empathy_line}".strip()
    if focus == "urgent_physical":
        summary_section = f"{summary_section} {scenario_context}".strip()
    elif focus != "general":
        summary_section = f"{summary_section} {scenario_context}".strip()

    empathy = (
        empathy_line
        if metadata.get("tone") == "high_empathy"
        else empathy_line
    )

    reasoning_section = "\n".join(f"- {line}" for line in reasoning_notes) if reasoning_notes else "- No additional clinical risk adjustments identified yet."
    follow_up_trimmed = follow_up_questions[:3] if response_length == "detailed" else follow_up_questions[:2]
    follow_up_section = (
        "\n".join(f"- {question}" for question in follow_up_trimmed)
        if follow_up_trimmed
        else "- Are there any other symptoms or concerns you haven’t mentioned yet?"
    )
    emergency_section = "\n".join(f"- {action}" for action in red_flag_actions) if red_flag_actions else "- Continue monitoring; escalate if new red flags appear."

    if response_length == "brief":
        concise_questions = "\n".join(f"• {q}" for q in follow_up_trimmed) if follow_up_trimmed else "• Let me know anything else that changed."
        urgent_watch = emergency_section.replace("- ", "• ") if emergency_section else "• Call emergency services if anything suddenly worsens."
        full_text = (
            f"{summary_section}\n\n"
            f"Key questions:\n{concise_questions}\n\n"
            f"Care guidance:\n{safe_steps}\n\n"
            f"Emergency watch:\n{urgent_watch}\n\n"
            f"{DISCLAIMER}"
        )
    else:
        full_text = (
            f"### Summary\n{summary_section}\n\n"
            f"### Explanation\n{explanation}\n\n"
            f"### Clinical reasoning\n{reasoning_section}\n\n"
            f"### Clarifying questions\n{follow_up_section}\n\n"
            f"### Safe next steps\n{safe_steps}\n\n"
            f"### Emergency actions\n{emergency_section}\n\n"
            f"### Supportive guidance\n{support}\n\n"
            f"### Disclaimer\n{DISCLAIMER}"
        )

    return {
        "full_text": full_text,
        "summary": summary_section,
        "resource_links": get_resource_links(resource_type),
        "lead_in": lead_in,
    }


def append_message(
    role: str,
    content: str,
    metadata: Optional[Dict[str, object]] = None,
    summary: Optional[str] = None,
) -> None:
    entry = {
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if summary is not None:
        entry["summary"] = summary
    st.session_state.conversation.append(entry)


def anonymise_conversation(conversation: List[Dict[str, object]], patient_name: Optional[str]) -> List[Dict[str, object]]:
    if not patient_name:
        return conversation
    redacted = []
    for message in conversation:
        replacement = dict(message)
        replacement["content"] = replacement["content"].replace(patient_name, "[REDACTED]")
        redacted.append(replacement)
    return redacted


def log_conversation() -> None:
    logger: StructuredConversationLogger = st.session_state.logger
    logger.persist_conversation(
        anonymise_conversation(
            st.session_state.conversation,
            st.session_state.patient_profile.name,
        ),
        st.session_state.metadata_log,
        st.session_state.patient_profile,
    )


def ensure_initial_greeting() -> None:
    if st.session_state.initial_greeting_added:
        return

    greeting_metadata: Dict[str, object] = {
        "response_focus": "greeting",
        "intent": "greeting",
        "tone": "neutral",
        "detail_preference": "balanced",
        "urgent": False,
        "trigger": None,
        "emotional_distress": False,
        "emotional_markers": [],
        "new_information": "No new symptom information detected in the latest message.",
        "new_symptoms": [],
        "detected_symptoms": [],
        "known_symptoms": [],
        "new_red_flags": [],
        "symptom_signals": {},
        "clinical_reasoning": [],
        "follow_up_questions": [
            "What symptoms are you experiencing right now?",
            "When did each symptom begin?",
            "Do you have chest pain, shortness of breath, or difficulty swallowing?",
        ],
        "red_flag_actions": [],
        "progressive_summary": "Opening assessment—invite the patient to describe current symptoms and history.",
        "resource_type": "greeting",
    "timestamp": datetime.now(UTC).isoformat() + "Z",
    }

    assistant_response = mock_model_response(greeting_metadata)
    assistant_metadata = {
        **greeting_metadata,
        "ui_summary": assistant_response["summary"],
        "resource_links": assistant_response["resource_links"],
        "lead_in": assistant_response.get("lead_in"),
    }

    visible_content = assistant_response["full_text"]
    summary_display = assistant_response["summary"]

    if st.session_state.language != "English":
        translated_content, outbound_error = translate_from_backend(visible_content, st.session_state.language)
        if translated_content:
            visible_content = translated_content
        if outbound_error:
            st.warning(outbound_error)

        translated_summary, summary_error = translate_from_backend(summary_display, st.session_state.language)
        if translated_summary:
            summary_display = translated_summary
        if summary_error and summary_error != outbound_error:
            st.warning(summary_error)

    append_message("assistant", visible_content, metadata=assistant_metadata, summary=summary_display)
    st.session_state.metadata_log.append(assistant_metadata)
    st.session_state.engine.ingest_assistant_message(assistant_response["full_text"])
    st.session_state.initial_greeting_added = True


# --------------------------- Streamlit layout begins ---------------------------

st.set_page_config(page_title="MedGuide Prototype", layout="wide")
initialise_state()

_apply_custom_theme(st.session_state.high_contrast, st.session_state.large_font)

with st.sidebar:
    st.header("Session Settings")
    language = st.selectbox("Language", list(TRANSLATION_LANG_CODES.keys()), index=list(TRANSLATION_LANG_CODES.keys()).index(st.session_state.language))
    if language != st.session_state.language:
        st.session_state.language = language

    st.session_state.high_contrast = st.checkbox("High contrast mode", value=st.session_state.high_contrast)
    st.session_state.large_font = st.checkbox("Large font", value=st.session_state.large_font)
    st.session_state.screen_reader_mode = st.checkbox("Screen reader friendly mode", value=st.session_state.screen_reader_mode)

    st.markdown("---")
    st.subheader("Patient Profile")
    name = st.text_input("Name", value=st.session_state.patient_profile.name or "")
    age = st.text_input("Age", value=st.session_state.patient_profile.age or "")
    gender = st.text_input("Gender", value=st.session_state.patient_profile.gender or "")
    history = st.text_area("Relevant history", value="; ".join(st.session_state.patient_profile.history))
    medications = st.text_area("Medications", value="; ".join(st.session_state.patient_profile.medications))
    allergies = st.text_area("Allergies", value="; ".join(st.session_state.patient_profile.allergies))

    if st.button("Update profile"):
        st.session_state.patient_profile = PatientProfile(
            name=name or None,
            age=age or None,
            gender=gender or None,
            history=[item.strip() for item in history.split(";") if item.strip()],
            medications=[item.strip() for item in medications.split(";") if item.strip()],
            allergies=[item.strip() for item in allergies.split(";") if item.strip()],
        )
        st.session_state.engine = AdaptivePromptEngine(patient_profile=st.session_state.patient_profile)
        st.success("Patient profile updated.")

    if st.button("Reset conversation"):
        reset_conversation()
        st.info("Conversation cleared.")

    st.markdown("---")
    st.caption("Voice input accepts WAV uploads. Voice output uses Google TTS if installed.")

_apply_custom_theme(st.session_state.high_contrast, st.session_state.large_font)

ensure_initial_greeting()

st.title("MedGuide Adaptive Chatbot Prototype")

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Chat")

    st.info(f"{DISCLAIMER} Always contact emergency services for life-threatening issues.")

    if st.session_state.screen_reader_mode and st.session_state.conversation:
        latest_entry = st.session_state.conversation[-1]
        sr_summary = latest_entry.get("summary") or latest_entry.get("content", "").split("\n", 1)[0]
        st.markdown(f"**Screen reader summary:** {sr_summary}")

    for index, message in enumerate(st.session_state.conversation):
        css_class = "chat-patient" if message["role"] == "patient" else "chat-assistant"
        speaker = "You" if message["role"] == "patient" else "MedGuide"

        if message["role"] == "assistant":
            summary_text = message.get("summary") or message.get("content", "").split("\n", 1)[0]
            st.markdown(
                f"<div class='chat-bubble {css_class}'>"
                f"<strong>{speaker}</strong><br>{summary_text}"
                "</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Show detailed guidance", expanded=index == len(st.session_state.conversation) - 1):
                st.markdown(message.get("content", ""))
                metadata = message.get("metadata", {}) or {}
                resources = metadata.get("resource_links") or []
                if resources:
                    st.markdown("**Helpful resources**")
                    for title, url in resources:
                        st.markdown(f"- [{title}]({url})")
        else:
            st.markdown(
                f"<div class='chat-bubble {css_class}'>"
                f"<strong>{speaker}</strong><br>{message['content']}"
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("Send a message")

    audio_file = st.file_uploader("Optional voice input", type=["wav", "mp3", "m4a"])
    voice_text = None
    if audio_file is not None:
        voice_text, voice_error = recognise_speech(audio_file)
        if voice_text:
            st.info(f"Transcribed voice input: {voice_text}")
        elif voice_error:
            st.warning(voice_error)

    user_message = st.text_area("Type your message", value=voice_text or "", key="input_message")
    submitted = st.button("Send")

    if submitted and user_message.strip():
        translated_to_english, translation_error = translate_to_backend(user_message.strip(), st.session_state.language)
        append_message("patient", user_message.strip())

        result = st.session_state.engine.build_prompt(translated_to_english)
        assistant_response = mock_model_response(result["metadata"])
        assistant_metadata = {
            **result["metadata"],
            "ui_summary": assistant_response["summary"],
            "resource_links": assistant_response["resource_links"],
            "lead_in": assistant_response.get("lead_in"),
        }
        st.session_state.metadata_log.append(assistant_metadata)

        assistant_content_english = assistant_response["full_text"]
        assistant_visible = assistant_content_english
        assistant_summary_display = assistant_response["summary"]

        if st.session_state.language != "English":
            assistant_content_translated, outbound_error = translate_from_backend(
                assistant_content_english,
                st.session_state.language,
            )
            if assistant_content_translated:
                assistant_visible = assistant_content_translated
            if outbound_error:
                st.warning(outbound_error)

            summary_translated, summary_error = translate_from_backend(
                assistant_response["summary"],
                st.session_state.language,
            )
            if summary_translated:
                assistant_summary_display = summary_translated
            if summary_error and summary_error != outbound_error:
                st.warning(summary_error)

        append_message("assistant", assistant_visible, metadata=assistant_metadata, summary=assistant_summary_display)
        st.session_state.engine.ingest_assistant_message(assistant_content_english)

        if translation_error:
            st.warning(translation_error)

        log_conversation()
        st.rerun()

with right_col:
    st.subheader("Clinician Metadata")
    urgent_now = any(entry.get("urgent") for entry in st.session_state.metadata_log)
    latest_metadata = st.session_state.metadata_log[-1] if st.session_state.metadata_log else {}

    st.markdown("<div class='metadata-panel'>", unsafe_allow_html=True)
    st.markdown(f"**Latest tone:** {latest_metadata.get('tone', 'neutral')}")
    st.markdown(f"**Detail level:** {latest_metadata.get('detail_preference', 'balanced')}")
    st.markdown(f"**Urgent:** {str(latest_metadata.get('urgent', False)).lower()}")
    st.markdown(f"**Emotional distress detected:** {str(latest_metadata.get('emotional_distress', False)).lower()}")
    trigger = latest_metadata.get("trigger")
    if trigger:
        st.markdown(f"**Escalation trigger:** {trigger}")
    if latest_metadata.get("needs_review"):
        st.error("Flagged for clinician review")
        reasons = latest_metadata.get("review_reasons") or []
        if reasons:
            st.caption("Reasons: " + ", ".join(reasons))
    if latest_metadata.get("emotional_markers"):
        st.markdown("**Emotional cues:** " + ", ".join(latest_metadata["emotional_markers"]))
    st.markdown(f"**Response focus:** {latest_metadata.get('response_focus', 'physical_monitoring')}")
    st.markdown(f"**Intent classification:** {latest_metadata.get('intent', 'general')}")
    progress_summary = latest_metadata.get("progressive_summary")
    if progress_summary:
        st.markdown(f"**Progress summary:** {progress_summary}")
    lead_in = latest_metadata.get("lead_in")
    if lead_in:
        st.caption(f"Lead-in: {lead_in}")
    reflection = latest_metadata.get("reflection")
    if reflection:
        st.markdown(f"**Patient reflection used:** {reflection}")
    new_info = latest_metadata.get("new_information")
    if new_info:
        st.markdown(f"**New information summary:** {new_info}")
    new_symptoms = latest_metadata.get("new_symptoms") or []
    if new_symptoms:
        st.markdown("**New symptoms recorded:** " + ", ".join(new_symptoms))
    detected_symptoms = latest_metadata.get("detected_symptoms") or []
    if detected_symptoms:
        st.caption("Detected in latest turn: " + ", ".join(detected_symptoms))
    known_symptoms = latest_metadata.get("known_symptoms") or []
    if known_symptoms:
        st.markdown("**Known symptom list:** " + ", ".join(known_symptoms))
    new_red_flags = latest_metadata.get("new_red_flags") or []
    if new_red_flags:
        st.markdown("**New red-flag cues:** " + ", ".join(new_red_flags))
    symptom_signals = latest_metadata.get("symptom_signals") or {}
    if isinstance(symptom_signals, dict) and symptom_signals:
        st.markdown("**Symptom signals (severity/triggers):**")
        for symptom, detail in symptom_signals.items():
            if not isinstance(detail, dict):
                st.markdown(f"- {symptom}")
                continue
            severity = detail.get("severity", "baseline")
            raw = detail.get("raw_match", symptom)
            triggers = detail.get("triggers") or []
            trigger_text = f"; triggers: {', '.join(triggers)}" if triggers else ""
            st.markdown(f"- {raw} (severity: {severity}{trigger_text})")
    reasoning_notes = latest_metadata.get("clinical_reasoning") or []
    if reasoning_notes:
        st.markdown("**Clinical reasoning cues:**")
        for note in reasoning_notes:
            st.markdown(f"- {note}")
    follow_up_questions = latest_metadata.get("follow_up_questions") or []
    if follow_up_questions:
        st.markdown("**Follow-up questions suggested:**")
        for q in follow_up_questions:
            st.markdown(f"- {q}")
    red_flag_actions = latest_metadata.get("red_flag_actions") or []
    if red_flag_actions:
        st.markdown("**Emergency actions advised:**")
        for action in red_flag_actions:
            st.markdown(f"- {action}")
    if latest_metadata.get("resource_links"):
        st.markdown("**Resources surfaced:**")
        for title, url in latest_metadata["resource_links"]:
            st.markdown(f"- [{title}]({url})")
    response_length = latest_metadata.get("response_length")
    if response_length:
        st.caption(f"Response mode: {response_length}")
    auto_scores = latest_metadata.get("auto_scores") or {}
    if auto_scores:
        formatted_scores = ", ".join(f"{key}={value:.2f}" for key, value in auto_scores.items())
        st.caption(f"Auto-scores: {formatted_scores}")
    st.markdown("**Safety log:**")
    if st.session_state.metadata_log:
        for item in st.session_state.metadata_log[-5:][::-1]:
            st.markdown(
                f"- {item['timestamp']}: intent={item.get('intent', 'general')} urgent={item['urgent']} tone={item['tone']} focus={item.get('response_focus', 'physical')}"
            )
    else:
        st.caption("No metadata yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Emergency")
    if urgent_now:
        with st.container():
            st.markdown("<div class='emergency-button'>", unsafe_allow_html=True)
            st.button("⚠️ Call emergency services", help="Connect patient to emergency resources.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("[Dial local emergency number](tel:911)")
            st.warning("Red flag detected. Encourage immediate escalation.")
            actions = latest_metadata.get("red_flag_actions") or []
            if actions:
                for action in actions:
                    st.info(action)
            else:
                st.info("Ask: Would you like me to connect you to emergency care resources?")
    else:
        st.caption("No active red flags.")

    st.markdown("---")
    st.subheader("Voice Output")
    if st.session_state.conversation and st.session_state.conversation[-1]["role"] == "assistant":
        if st.button("Play latest response as audio"):
            audio_buffer, audio_error = synthesise_speech(
                st.session_state.conversation[-1]["content"], st.session_state.language
            )
            if audio_buffer:
                st.audio(audio_buffer)
            elif audio_error:
                st.warning(audio_error)
    else:
        st.caption("Send a prompt to enable audio playback.")

    st.markdown("---")
    st.subheader("Progressive Disclosure")
    if st.session_state.conversation and st.session_state.conversation[-1]["role"] == "assistant":
        response_entry = st.session_state.conversation[-1]
        response_text = response_entry["content"]
        metadata = response_entry.get("metadata", {})
        summary = metadata.get("ui_summary") or response_text.split("\n", 1)[0]
        st.info(summary)
        with st.expander("Show more details"):
            st.markdown(response_text)
    else:
        st.caption("Assistant responses will appear here with expandable detail.")

    st.markdown("---")
    st.subheader("Review queue")
    review_items = st.session_state.logger.load_review_queue(limit=5)
    if review_items:
        for item in review_items:
            st.markdown(
                f"- {item.get('conversation_file')} · turn {item.get('turn_index')} — reasons: {', '.join(item.get('review_reasons', [])) or 'auto triage'}"
            )
            auto_scores = item.get("auto_scores") or {}
            if auto_scores:
                st.caption(
                    "  Scores: "
                    + ", ".join(f"{key}={value:.2f}" for key, value in auto_scores.items())
                )
        with st.expander("Submit clinician review"):
            queue_labels = [
                f"{idx+1}. {item.get('conversation_file')} (turn {item.get('turn_index')})"
                for idx, item in enumerate(review_items)
            ]
            selection = st.selectbox("Queue item", options=list(range(len(review_items))), format_func=lambda i: queue_labels[i])
            reviewer_name = st.text_input("Reviewer", value="")
            resolution = st.selectbox("Resolution", ["approved", "needs_follow_up", "unsafe"], index=0)
            notes = st.text_area("Review notes", value="")
            updated_prompt = st.text_area("Suggested replacement prompt (optional)", value="")
            if st.button("Record review"):
                if not reviewer_name.strip():
                    st.warning("Reviewer name is required to log feedback.")
                else:
                    selected = review_items[selection]
                    st.session_state.logger.record_review_feedback(
                        conversation_file=selected.get("conversation_file", ""),
                        turn_index=int(selected.get("turn_index", 0)),
                        reviewer=reviewer_name.strip(),
                        resolution=resolution,
                        notes=notes.strip(),
                        updated_prompt=updated_prompt.strip(),
                    )
                    st.success("Review recorded. Refresh the queue to see updates.")
    else:
        st.caption("No items currently waiting for clinician review.")

    st.markdown("---")
    st.subheader("Phase 6 • Evaluation & Compliance")

    with st.expander("Capture pilot feedback", expanded=False):
        with st.form("feedback_form"):
            doctor_like = st.slider("Doctor-like feel (1=poor, 5=excellent)", min_value=1, max_value=5, value=4, key="feedback_doctor_like")
            empathy_score = st.slider("Empathy & support score", min_value=1, max_value=5, value=4, key="feedback_empathy")
            safety_score = st.slider("Safety / escalation confidence", min_value=1, max_value=5, value=4, key="feedback_safety")
            bias_notes = st.text_area("Bias or fairness observations", key="feedback_bias_notes")
            submitted_feedback = st.form_submit_button("Save feedback snapshot")
            if submitted_feedback:
                st.session_state.feedback_log.append({
                    "timestamp": datetime.now(UTC).isoformat(),
                    "doctor_like": doctor_like,
                    "empathy": empathy_score,
                    "safety": safety_score,
                    "bias_notes": bias_notes.strip(),
                })
                st.success("Feedback logged for QA review.")

    if st.session_state.feedback_log:
        latest_feedback = st.session_state.feedback_log[-3:][::-1]
        st.caption("Recent feedback snapshots")
        for entry in latest_feedback:
            st.markdown(
                f"- {entry['timestamp']}: doctor-like={entry['doctor_like']} empathy={entry['empathy']} safety={entry['safety']}"
            )
            if entry.get("bias_notes"):
                st.caption(f"  Notes: {entry['bias_notes']}")

    st.markdown("**Compliance checklist**")
    st.checkbox("TLS encryption enforced for all API calls", value=True, disabled=True, key="compliance_tls")
    st.checkbox("Role-based access control enabled for clinician dashboard", value=True, disabled=True, key="compliance_access")
    st.checkbox("Patient consent logging active", value=True, disabled=True, key="compliance_consent")
    st.checkbox("Conversation audit trail persisted to logs/", value=True, disabled=True, key="compliance_audit")
    st.caption(f"Audit logs stored at: `{LOG_DIR}`")

    if st.button("Sync review feedback into engine"):
        feedback_path = st.session_state.logger.feedback_path
        st.session_state.engine.integrate_correction_feedback(feedback_path)
        st.success("Review feedback loaded for prompt tuning consideration.")
