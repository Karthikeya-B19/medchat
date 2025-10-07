"""FastAPI server exposing the MedGuide adaptive prompt engine."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.phase5.adaptive_prompt_engine import AdaptivePromptEngine, PatientProfile

app = FastAPI(title="MedGuide Chatbot", version="0.1.0")


class ProfilePayload(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    history: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)

    def to_profile(self) -> PatientProfile:
        return PatientProfile(
            name=self.name,
            age=self.age,
            gender=self.gender,
            history=self.history,
            medications=self.medications,
            allergies=self.allergies,
        )


class ConversationTurnPayload(BaseModel):
    speaker: Literal["patient", "assistant"]
    text: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="Latest patient utterance to process")
    conversation: List[ConversationTurnPayload] = Field(
        default_factory=list,
        description="Optional prior conversation turns to rebuild state",
    )
    profile: ProfilePayload = Field(default_factory=ProfilePayload)
    include_prompt: bool = Field(True, description="Return the structured prompt for downstream LLM calls")


class ChatResponse(BaseModel):
    metadata: Dict[str, object]
    prompt: Optional[str] = None
    triage_actions: List[str] = Field(default_factory=list)
    needs_review: bool = False


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    engine = AdaptivePromptEngine(patient_profile=request.profile.to_profile())

    # Rehydrate context with prior turns (if any)
    for turn in request.conversation:
        engine.state.add_turn(turn.speaker, turn.text)

    result = engine.build_prompt(request.message)
    metadata: Dict[str, object] = result["metadata"]
    prompt: Optional[str] = result["prompt"] if request.include_prompt else None

    triage_actions = metadata.get("red_flag_actions") or []
    if metadata.get("urgent") and not triage_actions:
        triage_actions = [
            "Emergency suspicion raised—call local emergency services immediately.",
        ]

    return ChatResponse(
        metadata=metadata,
        prompt=prompt,
        triage_actions=list(triage_actions),
        needs_review=bool(metadata.get("needs_review")),
    )
