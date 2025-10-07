# Phase 5 Prototype UI

Interactive Streamlit prototype that wraps the adaptive prompting engine for the MedGuide assistant.

## Features
- Patient-facing chat window with a doctor-style greeting, empathetic tone, and progressive disclosure (summary bubble first, expandable details).
- Adaptive follow-up question generation that highlights missing severity/timing details and always asks about chest pain, shortness of breath, or swallowing difficulty.
- Clinical reasoning cues, red-flag categorisation, and emergency actions surfaced alongside resource recommendations.
- Clinician metadata panel showing tone, intent, urgency, reasoning notes, follow-up prompts, and recent safety logs.
- Emergency escalation button, tel:911 quick link, and dynamic guidance when urgent flags are detected.
- Accessibility and language support: high-contrast theme, large fonts, screen-reader-friendly mode, multilingual translation (English, Hindi, Tamil, Telugu, Kannada), and optional voice input/output hooks when the dependencies are installed.
- Automatic logging of each conversation with simple anonymisation of the patient name plus a compliance dashboard (feedback capture, TLS/access/consent checkboxes).
- Structured logging pipeline writes JSON logs with auto-scored safety/empathy metrics and triages low-confidence turns into a review queue.
- Clinician review console surfaces triage items and lets reviewers resolve cases or attach corrected prompts; feedback syncs back into the engine for guardrail tuning.
- Companion FastAPI service (`app/api/server.py`) exposes the adaptive prompt builder for deployment workflows.

## Running the prototype
```
streamlit run app/phase5/ui_app.py
```

Optional dependencies for enhanced functionality:
- `deep-translator` – translation support
- `SpeechRecognition` – speech-to-text for audio uploads
- `gTTS` – text-to-speech audio playback
- `fastapi` / `uvicorn` – API deployment endpoint

Install them with:
```
pip install streamlit deep-translator SpeechRecognition gTTS
```

## Next steps
- Swap the `mock_model_response` placeholder with the fine-tuned MedGuide model inference pipeline.
- Persist logs to a secure database and extend anonymisation to additional identifiers.
- Add clinician authentication to gate access to the metadata panel and review console.
- Wire the Phase 6 feedback log into analytics (doctor-like feel, empathy, safety, bias) and surface aggregate metrics.
- Integrate with deployment tooling (AWS/GCP/Azure) using encrypted transport and configurable emergency contact routing.
- Extend the retraining scheduler with automated dataset curation and CI hooks.

## API quick start

Launch the FastAPI deployment target:

```
uvicorn app.api.server:app --reload
```

Send a request containing the latest patient message plus optional history:

```
POST /chat
{
	"message": "I still have a fever and now my neck is stiff.",
	"conversation": [
		{"speaker": "patient", "text": "I've had a fever for two days."}
	]
}
```

The response returns the structured prompt and metadata (including auto-scores and red-flag actions) so downstream services can select the right model or escalate.

## Retraining workflow

- Clinicians capture corrections via the review console; feedback is appended to `app/phase5/logs/review_feedback.jsonl`.
- Run `python app/phase6/retraining.py` to generate a manifest of unsafe or follow-up cases ready for fine-tuning.
- Use the manifest to schedule weekly guardrail reviews or active-learning batches.
