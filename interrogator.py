"""
Interrogator Agent — Part 1 of Project OMNISCIENT.

Responsibilities:
  - Frame questions and conduct multi-turn interrogation sessions with suspect agents.
  - Extract structured Claim objects from raw suspect responses via an LLM parser.
  - Decide whether a follow-up question is warranted based on claim quality.
  - Maintain persona stability by embedding the detective persona in the system
    prompt at session start rather than injecting runtime instructions.

Communication protocol:
  - Receives:  InterrogationGoal (from ReasoningEngine.next_goal())
  - Returns:   Testimony (consumed by ReasoningEngine.ingest_testimony())

Token tracking:
  - All LLM calls route through self._counter.add(response.usage) so the
    investigation's total token spend is recorded in one shared TokenCounter.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from openai import OpenAI

from token_counter import TokenCounter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ClaimType(str, Enum):
    """Controlled vocabulary for the semantic type of an extracted claim."""

    ALIBI = "alibi"                    # suspect asserts their own location/activity
    ACCUSATION = "accusation"          # suspect implicates another person
    DENIAL = "denial"                  # suspect denies an action or presence
    RELATIONSHIP = "relationship"      # suspect describes connection to victim/others
    KNOWLEDGE = "knowledge"            # suspect reveals what they know about the crime
    INCONSISTENCY = "inconsistency"    # parser detected internal contradiction in answer


class Persona(str, Enum):
    """
    Interrogation personas available to the detective.

    Each persona shapes question framing and linguistic register.
    Persona stability is enforced by embedding the persona description
    in the system prompt once at session start (build_interrogator_system_prompt),
    rather than injecting runtime instructions such as "You are now hostile."
    Embedding at construction time is stable over arbitrarily long contexts
    because it anchors the model's generation distribution from the first token.
    """

    ANALYTICAL = "analytical"    # logical, precise, evidence-focused
    SYMPATHETIC = "sympathetic"  # warm, empathetic, rapport-building
    HOSTILE = "hostile"          # confrontational, pressure-applying


class ObjectiveType(str, Enum):
    """The investigative goal driving a particular interrogation session."""

    ESTABLISH_ALIBI = "establish_alibi"
    PROBE_MOTIVE = "probe_motive"
    VERIFY_RELATIONSHIP = "verify_relationship"
    CONFRONT_CONFLICT = "confront_conflict"    # directly probe a known contradiction
    OPEN_ENDED = "open_ended"                 # initial sweep, no specific target


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """One atomic factual claim extracted from a suspect's answer."""

    claim_id: str
    claim_type: ClaimType
    subject: str
    predicate: str
    object_: str
    confidence: str           # "stated" | "implied" | "uncertain"
    source_text: str          # verbatim fragment grounding this claim
    time_ref: Optional[str] = None


@dataclass
class InterrogationGoal:
    """
    Instruction packet sent from the ReasoningEngine to the Interrogator.
    Encapsulates everything the Interrogator needs to conduct one session.
    """

    suspect_name: str
    objective: ObjectiveType
    persona: Persona = Persona.ANALYTICAL
    time_window: Optional[str] = None      # e.g. "9PM-11PM"
    known_conflicts: list = field(default_factory=list)  # plain-English discrepancies
    max_turns: int = 2                     # conversation budget per session


@dataclass
class Testimony:
    """
    Structured output produced by the Interrogator after one session.
    This is the unit of information consumed by the ReasoningEngine.
    """

    testimony_id: str
    suspect_name: str
    objective: ObjectiveType
    persona_used: Persona
    turns: list            # list of {turn, question, raw_answer, claims_extracted}
    claims: list           # list of Claim objects
    emotional_tone: str = "neutral"       # cooperative | neutral | guarded | evasive
    evasiveness_score: int = 0            # 0-5 scale derived by session assessor
    follow_up_warranted: bool = False
    follow_up_reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    parse_error: bool = False
    parse_error_detail: str = ""

    def to_dict(self) -> dict:
        """Serialise to a plain dict for logging and report generation."""
        return {
            "testimony_id": self.testimony_id,
            "suspect_name": self.suspect_name,
            "objective": self.objective.value,
            "persona_used": self.persona_used.value,
            "turns": self.turns,
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "claim_type": c.claim_type.value,
                    "subject": c.subject,
                    "predicate": c.predicate,
                    "object_": c.object_,
                    "confidence": c.confidence,
                    "source_text": c.source_text,
                    "time_ref": c.time_ref,
                }
                for c in self.claims
            ],
            "emotional_tone": self.emotional_tone,
            "evasiveness_score": self.evasiveness_score,
            "follow_up_warranted": self.follow_up_warranted,
            "follow_up_reason": self.follow_up_reason,
            "timestamp": self.timestamp,
            "parse_error": self.parse_error,
            "parse_error_detail": self.parse_error_detail,
        }


# ---------------------------------------------------------------------------
# Static prompts — defined at module level so generate_training_data.py
# can import PARSER_SYSTEM_PROMPT and remain in sync with inference time.
# ---------------------------------------------------------------------------

INTERROGATOR_PERSONA_PROMPTS: dict = {
    Persona.ANALYTICAL: (
        "You are a precise, methodical detective. Your questions are concise and "
        "evidence-focused. You do not speculate. You seek verifiable facts: "
        "locations, times, witnesses, physical evidence. You do not express emotion. "
        "If an answer is vague, you ask for specifics. Never ask compound questions."
    ),
    Persona.SYMPATHETIC: (
        "You are a warm, empathetic detective. You build rapport before probing. "
        "You acknowledge the suspect's feelings before asking hard questions. "
        "Your tone is conversational, never accusatory. You listen carefully and "
        "reflect back what you hear before asking follow-ups."
    ),
    Persona.HOSTILE: (
        "You are a confrontational detective. You apply pressure. You challenge "
        "inconsistencies directly. You do not accept vague answers. You imply "
        "you already know more than you are revealing. Your questions are sharp "
        "and pointed. You never threaten — you simply make clear you are not "
        "convinced."
    ),
}

PARSER_SYSTEM_PROMPT: str = (
    "You are a forensic claim extractor. You receive a question and a suspect's "
    "raw answer. Extract every factual claim as a JSON array.\n\n"
    "Each element must have these exact keys:\n"
    "  claim_type : one of alibi | accusation | denial | relationship | knowledge | inconsistency\n"
    "  subject    : who the claim is about (full name)\n"
    "  predicate  : the relationship or action (snake_case verb phrase)\n"
    "  object_    : what the predicate points to\n"
    "  time_ref   : time reference string or null\n"
    "  confidence : stated | implied | uncertain\n"
    "  source_text: the exact fragment from the answer that grounds this claim\n\n"
    "Return ONLY a valid JSON array. No preamble, no explanation, no markdown fences."
)


# ---------------------------------------------------------------------------
# Interrogator class
# ---------------------------------------------------------------------------

class Interrogator:
    """
    Frames questions, conducts multi-turn sessions, and extracts structured claims.

    Uses two separate LLM roles:
      - detective_client/model: generates questions in the chosen persona.
      - parser_client/model:    extracts structured Claim objects from raw answers.

    Both roles can use the same client and model; they are separated so that
    a fine-tuned local parser model can be plugged in independently of the
    question-generation model.

    Args:
        detective_client : OpenAI-compatible client for question generation.
        detective_model  : Model identifier for the detective role.
        parser_client    : OpenAI-compatible client for claim extraction.
        parser_model     : Model identifier for the parser role.
        counter          : Shared TokenCounter; records all API token usage.
    """

    def __init__(
        self,
        detective_client: OpenAI,
        detective_model: str,
        parser_client: OpenAI,
        parser_model: str,
        counter: TokenCounter,
    ) -> None:
        self.detective_client = detective_client
        self.detective_model = detective_model
        self.parser_client = parser_client
        self.parser_model = parser_model
        self._counter = counter

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _objective_to_instruction(self, goal: InterrogationGoal) -> str:
        """Map an ObjectiveType to a plain-English interrogation instruction."""
        mapping = {
            ObjectiveType.ESTABLISH_ALIBI: (
                f"Establish a clear, verifiable alibi for {goal.suspect_name}. "
                f"Focus on the time window: {goal.time_window or 'the time of the crime'}."
            ),
            ObjectiveType.PROBE_MOTIVE: (
                f"Uncover possible motives {goal.suspect_name} may have had "
                "against the victim. Probe recent tensions, disputes, or grievances."
            ),
            ObjectiveType.VERIFY_RELATIONSHIP: (
                f"Clarify {goal.suspect_name}'s relationship with the victim. "
                "Establish how well they knew each other and the quality of that relationship."
            ),
            ObjectiveType.CONFRONT_CONFLICT: (
                "Directly confront the suspect with the known discrepancies listed below. "
                "Press for a concrete explanation."
            ),
            ObjectiveType.OPEN_ENDED: (
                "Conduct an open-ended initial interview. Gather as much context as possible "
                "about the suspect's knowledge of the victim and the events of that evening."
            ),
        }
        return mapping.get(goal.objective, "Ask relevant questions about the case.")

    def build_interrogator_system_prompt(self, goal: InterrogationGoal) -> str:
        """
        Construct the detective system prompt for this session.

        The persona description is embedded here at session construction rather
        than injected as a runtime instruction. This ensures the persona remains
        stable across the full session context without attention dilution.
        """
        base = INTERROGATOR_PERSONA_PROMPTS[goal.persona]
        objective_instruction = self._objective_to_instruction(goal)

        conflict_block = ""
        if goal.known_conflicts:
            formatted = "\n".join(f"  - {c}" for c in goal.known_conflicts)
            conflict_block = (
                f"\n\nKNOWN CONFLICTS TO PROBE:\n{formatted}\n"
                "Address these discrepancies directly when appropriate."
            )

        time_block = ""
        if goal.time_window:
            time_block = (
                f"\n\nFOCUS TIME WINDOW: {goal.time_window}. "
                "Establish what the suspect was doing during this period."
            )

        return (
            f"{base}\n\n"
            f"YOUR CURRENT OBJECTIVE: {objective_instruction}"
            f"{time_block}"
            f"{conflict_block}\n\n"
            "Ask one question at a time. Be direct. Stay in character."
        )

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _generate_question(
        self,
        interrogator_history: list,
        goal: InterrogationGoal,
        is_opening: bool,
        follow_up_reason: Optional[str] = None,
    ) -> str:
        """
        Ask the detective LLM to produce exactly one question.

        Args:
            interrogator_history : Running message list for the detective role.
            goal                 : Current session goal.
            is_opening           : True for the first question, False for follow-ups.
            follow_up_reason     : Optional hint about why a follow-up is needed
                                   (e.g. "no_claims_extracted", "uncertain_claims").
                                   Passed only on non-opening calls; instructs the
                                   detective to target the specific gap rather than
                                   repeating the previous question.

        Returns:
            Question text as a plain string.
        """
        if is_opening:
            user_msg = (
                f"Begin the interrogation of {goal.suspect_name}. "
                "Generate your opening question. Return only the question text, nothing else."
            )
        else:
            # Tailor the follow-up instruction by reason so the detective targets
            # the specific deficiency in the previous answer instead of repeating
            # the previous question verbatim.
            reason_hint = ""
            if follow_up_reason == "no_claims_extracted":
                reason_hint = (
                    " The previous answer contained no verifiable facts. "
                    "Ask a sharper, more specific question that forces a concrete "
                    "factual response (a name, a place, a time)."
                )
            elif follow_up_reason and follow_up_reason.startswith("uncertain_claims"):
                reason_hint = (
                    " The previous answer contained uncertain or hedged claims. "
                    "Ask a clarifying question that pins down the uncertainty — "
                    "request a specific name, exact time, or independent witness."
                )
            elif follow_up_reason:
                reason_hint = f" Follow-up reason: {follow_up_reason}."

            user_msg = (
                "Based on the conversation so far, generate your NEXT question. "
                "It must be different from any question you have already asked."
                f"{reason_hint}"
                " Return only the question text, nothing else."
            )

        messages = interrogator_history + [{"role": "user", "content": user_msg}]

        response = self.detective_client.chat.completions.create(
            model=self.detective_model,
            messages=messages,
            temperature=0.4,
            max_tokens=80,
        )
        self._counter.add(response.usage)
        question = response.choices[0].message.content.strip()
        logger.debug("Generated question for %s: %s", goal.suspect_name, question)
        return question

    def _extract_claims(
        self,
        question: str,
        raw_answer: str,
        suspect_name: str,
    ) -> list:
        """
        Run the parser LLM on a question-answer pair and return Claim objects.

        Falls back to an empty list on JSON parse failure or API error;
        the parse_error flag on the Testimony is set in conduct_session.

        Args:
            question     : The question that was asked.
            raw_answer   : The suspect's raw text response.
            suspect_name : Used to populate claim subjects and for logging.

        Returns:
            List of Claim objects (may be empty on failure).
        """
        user_content = (
            f"SUSPECT NAME: {suspect_name}\n"
            f"QUESTION: {question}\n"
            f"ANSWER: {raw_answer}"
        )

        try:
            response = self.parser_client.chat.completions.create(
                model=self.parser_model,
                messages=[
                    {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.0,
                max_tokens=400,
            )
            self._counter.add(response.usage)
            raw_json = response.choices[0].message.content.strip()

            # Strip markdown fences if the model returned them despite the prompt
            if raw_json.startswith("```"):
                raw_json = raw_json.split("```")[1]
                if raw_json.startswith("json"):
                    raw_json = raw_json[4:]

            parsed = json.loads(raw_json)

        except json.JSONDecodeError as exc:
            logger.warning(
                "Claim extraction JSON parse failed | suspect=%s | error=%s",
                suspect_name, exc,
            )
            return []
        except Exception as exc:
            logger.error("Claim extraction API error | suspect=%s | error=%s", suspect_name, exc)
            return []

        claims = []
        for item in parsed:
            try:
                claim = Claim(
                    claim_id=str(uuid.uuid4())[:8],
                    claim_type=ClaimType(item["claim_type"]),
                    subject=item["subject"],
                    predicate=item["predicate"],
                    object_=item["object_"],
                    time_ref=item.get("time_ref"),
                    confidence=item.get("confidence", "stated"),
                    source_text=item.get("source_text", ""),
                )
                claims.append(claim)
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed claim: %s | error=%s", item, exc)

        logger.debug(
            "Extracted %d claims from %s's answer", len(claims), suspect_name
        )
        return claims

    def _decide_follow_up(
        self,
        interrogator_history: list,
        goal: InterrogationGoal,
        last_answer: str,
        last_claims: list,
    ) -> dict:
        """
        Determine whether a follow-up question is warranted after the last turn.

        Decision hierarchy:
          1. If no claims extracted → answer was likely evasive; follow up.
          2. If any claim has confidence=="uncertain" → press for clarity.
          3. Otherwise, ask the detective LLM to decide (covers nuanced cases).

        Returns:
            Dict with keys: warranted (bool), question (str), reason (str).
        """
        # Rule 1: no claims extracted means the answer contained no actionable facts
        if not last_claims:
            question = self._generate_question(
                interrogator_history, goal,
                is_opening=False,
                follow_up_reason="no_claims_extracted",
            )
            return {
                "warranted": True,
                "question": question,
                "reason": "no_claims_extracted",
            }

        # Rule 2: uncertain claims need clarification
        uncertain = [c for c in last_claims if c.confidence == "uncertain"]
        if uncertain:
            reason_str = f"uncertain_claims:{[c.claim_id for c in uncertain]}"
            question = self._generate_question(
                interrogator_history, goal,
                is_opening=False,
                follow_up_reason=reason_str,
            )
            return {
                "warranted": True,
                "question": question,
                "reason": reason_str,
            }

        # Rule 3: deterministic default — no extra LLM call.
        # If neither Rule 1 (no claims) nor Rule 2 (uncertain claims) fired,
        # the previous turn produced concrete, confident claims and another
        # follow-up would yield diminishing returns. Skipping the LLM-based
        # decision here saves one call per turn that didn't follow up.
        return {"warranted": False, "question": "", "reason": "rules_satisfied"}

    # ------------------------------------------------------------------
    # Session assessment
    # ------------------------------------------------------------------

    def _assess_session(self, turns: list) -> tuple:
        """
        Derive an emotional tone label and evasiveness score from the session turns.

        Evasiveness scoring (0-5):
          +1 per turn with < 20 words           (terse = guarded)
          +1 per turn with zero claims          (no actionable facts surfaced)
          +1 per turn where ALL claims are uncertain
                                                (hedged, deflective answers)
          +1 if session needed the maximum
              number of turns                   (suspect resisted to the end)
        Capped at 5.

        Tone is determined by multiple signals, not just length. A short
        cooperative answer with multiple concrete claims is NOT guarded.
        Decision order (first match wins):
          1. Any turn with no claims                       -> evasive
          2. Average words < 15                            -> evasive
          3. Average words < 20 with no claim density      -> guarded
          4. All claims across the session are uncertain   -> guarded
          5. Multiple concrete claims per turn on average  -> cooperative
          6. Average words >= 25 with no red flags         -> cooperative
          7. Otherwise                                     -> neutral

        Args:
            turns : List of turn dicts from conduct_session.

        Returns:
            (emotional_tone: str, evasiveness_score: int)
        """
        if not turns:
            return "neutral", 0

        total_words = 0
        total_claims = 0
        evasiveness = 0
        short_turns = 0
        zero_claim_turns = 0
        all_uncertain_turns = 0

        for turn in turns:
            answer = turn.get("raw_answer", "") or ""
            word_count = len(answer.split())
            total_words += word_count

            claims_in_turn = turn.get("claims_extracted", 0)
            total_claims += claims_in_turn

            if word_count < 20:
                short_turns += 1
                evasiveness += 1

            if claims_in_turn == 0:
                zero_claim_turns += 1
                evasiveness += 1

            # Track turns where every claim extracted was tagged uncertain.
            # The turn dict only stores the COUNT of claims, not their confidence,
            # so this signal is approximated using "uncertain_claims" field if
            # the Interrogator chose to record it. Falls back to 0 if absent.
            uncertain_in_turn = turn.get("uncertain_claims", 0)
            if claims_in_turn > 0 and uncertain_in_turn == claims_in_turn:
                all_uncertain_turns += 1
                evasiveness += 1

        if len(turns) >= 3:
            evasiveness += 1

        evasiveness = min(evasiveness, 5)

        avg_words = total_words / max(len(turns), 1)
        claim_density = total_claims / max(len(turns), 1)

        # Tone decision order
        if zero_claim_turns > 0:
            tone = "evasive"
        elif avg_words < 15:
            tone = "evasive"
        elif avg_words < 20 and claim_density < 1.5:
            tone = "guarded"
        elif all_uncertain_turns == len(turns):
            tone = "guarded"
        elif claim_density >= 2.0:
            tone = "cooperative"
        elif avg_words >= 25 and short_turns == 0 and all_uncertain_turns == 0:
            tone = "cooperative"
        else:
            tone = "neutral"

        return tone, evasiveness

    # ------------------------------------------------------------------
    # Main session entry point
    # ------------------------------------------------------------------

    def conduct_session(
        self,
        goal: InterrogationGoal,
        suspect_agent,
    ) -> Testimony:
        """
        Conduct one complete interrogation session against a single suspect agent.

        Flow:
          1. Build the detective system prompt embedding persona and objective.
          2. Generate opening question.
          3. Loop: ask question → get answer → extract claims → decide follow-up.
          4. Assess session tone and evasiveness.
          5. Return a structured Testimony.

        Args:
            goal          : InterrogationGoal from the ReasoningEngine.
            suspect_agent : SuspectAgent instance with an .interrogate(question) method.

        Returns:
            Testimony object consumed by ReasoningEngine.ingest_testimony().
        """
        testimony_id = str(uuid.uuid4())[:8]
        logger.info(
            "Session START | id=%s | suspect=%s | objective=%s | persona=%s | max_turns=%d",
            testimony_id,
            goal.suspect_name,
            goal.objective.value,
            goal.persona.value,
            goal.max_turns,
        )

        turns: list = []
        claims: list = []
        parse_error = False
        parse_error_detail = ""

        # The detective system prompt is built once and never mutated during the session.
        # This ensures persona stability over the full context of the interrogation.
        system_prompt = self.build_interrogator_system_prompt(goal)
        interrogator_history = [{"role": "system", "content": system_prompt}]

        current_question = self._generate_question(
            interrogator_history, goal, is_opening=True
        )

        turns_taken = 0

        while turns_taken < goal.max_turns:
            logger.debug(
                "Turn %d/%d | suspect=%s | question=%s",
                turns_taken + 1,
                goal.max_turns,
                goal.suspect_name,
                current_question,
            )

            raw_answer = suspect_agent.interrogate(current_question)

            turn_claims = self._extract_claims(
                question=current_question,
                raw_answer=raw_answer,
                suspect_name=goal.suspect_name,
            )

            if not turn_claims and raw_answer:
                # Record extraction failure for anomaly reporting
                parse_error = True
                parse_error_detail = (
                    f"Turn {turns_taken + 1}: parser returned 0 claims for a non-empty answer."
                )

            claims.extend(turn_claims)

            # Count uncertain claims in this turn so _assess_session can use
            # this as a signal for hedged/deflective answers.
            uncertain_in_turn = sum(
                1 for c in turn_claims if c.confidence == "uncertain"
            )

            turn_record = {
                "turn": turns_taken + 1,
                "question": current_question,
                "raw_answer": raw_answer,
                "claims_extracted": len(turn_claims),
                "uncertain_claims": uncertain_in_turn,
            }
            turns.append(turn_record)

            # Update detective conversation history for context continuity
            interrogator_history.append(
                {"role": "assistant", "content": current_question}
            )
            interrogator_history.append(
                {"role": "user", "content": f"[Suspect answered]: {raw_answer}"}
            )

            turns_taken += 1

            # Decide whether to continue before consuming another turn budget
            if turns_taken < goal.max_turns:
                follow_up = self._decide_follow_up(
                    interrogator_history, goal, raw_answer, turn_claims
                )
                if follow_up.get("warranted"):
                    current_question = follow_up["question"]
                    logger.debug(
                        "Follow-up warranted | reason=%s | next_question=%s",
                        follow_up.get("reason"),
                        current_question,
                    )
                else:
                    logger.debug(
                        "No follow-up warranted | reason=%s | closing session early",
                        follow_up.get("reason"),
                    )
                    break

        emotional_tone, evasiveness_score = self._assess_session(turns)

        testimony = Testimony(
            testimony_id=testimony_id,
            suspect_name=goal.suspect_name,
            objective=goal.objective,
            persona_used=goal.persona,
            turns=turns,
            claims=claims,
            emotional_tone=emotional_tone,
            evasiveness_score=evasiveness_score,
            follow_up_warranted=False,
            follow_up_reason="",
            parse_error=parse_error,
            parse_error_detail=parse_error_detail,
        )

        logger.info(
            "Session END | id=%s | turns=%d | claims=%d | evasiveness=%d | tone=%s",
            testimony_id,
            turns_taken,
            len(claims),
            evasiveness_score,
            emotional_tone,
        )

        return testimony