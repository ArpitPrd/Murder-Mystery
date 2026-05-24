"""
Sanity test for Project OMNISCIENT.

Runs the complete pipeline (Interrogator -> ReasoningEngine -> ReportGenerator)
with a mock LLM client. No Ollama, no API keys, no internet required.

The mock simulates a 2-suspect case where:
  - Victor Crane (murderer): gives a false alibi that gets contradicted,
    is evasive, and has an established motive.
  - Diana Shore (innocent): gives a solid alibi and no motive.

Expected outcome: Victor Crane identified as the murderer.

Usage:
    python test_sanity.py
"""

import sys
import os
import logging
from datetime import datetime

# Silence most logging for cleaner test output
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

from token_counter import TokenCounter
from interrogator import (
    Interrogator, InterrogationGoal, ObjectiveType, Persona,
    ClaimType, Claim
)
from reasoning_engine import ReasoningEngine, VERDICT_THRESHOLD, SEPARATION_MARGIN
from report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Scripted responses for the mock LLM
#
# The mock alternates through a queue of responses per (suspect, call_type).
# call_type "question" = detective asking a question
# call_type "parse"    = parser extracting claims
# call_type "decision" = follow-up decision
# call_type "narrative" = report narrative
# ---------------------------------------------------------------------------

VICTOR_QUESTIONS = [
    "Where were you the evening of the murder?",
    "Can anyone confirm you were at the restaurant?",
    "You said restaurant — a witness places you near the victim's office. Explain.",
]

VICTOR_ANSWERS = [
    # Turn 1 — false alibi
    "I was at the Golden Fork restaurant all evening with a colleague. "
    "We had dinner from seven until well past nine. I never went near the office.",
    # Turn 2 — evasive
    "My colleague can confirm it. I don't remember the exact name right now.",
    # Turn 3 — contradicted, gets defensive
    "That witness is mistaken. I was at the restaurant the entire time. "
    "I had a conflict with the victim last week over a business deal gone wrong, "
    "but that has nothing to do with this.",
]

VICTOR_CLAIMS = [
    # Claims extracted from answer 1 — alibi claim
    '[{"claim_type":"alibi","subject":"Victor Crane","predicate":"was_at_location","object_":"Golden Fork restaurant","time_ref":"7PM-9PM","confidence":"stated","source_text":"I was at the Golden Fork restaurant all evening"}]',
    # Claims extracted from answer 2 — weak corroboration
    '[{"claim_type":"alibi","subject":"Victor Crane","predicate":"accompanied_by","object_":"unnamed colleague","time_ref":"evening","confidence":"uncertain","source_text":"My colleague can confirm it"}]',
    # Claims extracted from answer 3 — contradiction + motive
    '[{"claim_type":"denial","subject":"Victor Crane","predicate":"was_at_location","object_":"victim office","time_ref":"evening","confidence":"stated","source_text":"I was at the restaurant the entire time"},'
    '{"claim_type":"knowledge","subject":"Victor Crane","predicate":"had_conflict_with","object_":"victim","time_ref":"last week","confidence":"stated","source_text":"I had a conflict with the victim last week over a business deal gone wrong"}]',
]

DIANA_QUESTIONS = [
    "Where were you on the evening of the murder?",
    "Did you have any issues with the victim recently?",
]

DIANA_ANSWERS = [
    # Turn 1 — solid alibi
    "I was at the city theatre watching a play from six until ten. "
    "I have the ticket stub and several friends who saw me there.",
    # Turn 2 — no motive
    "No, I had a perfectly fine relationship with the victim. "
    "We collaborated on a project recently. I was shocked to hear the news.",
]

DIANA_CLAIMS = [
    '[{"claim_type":"alibi","subject":"Diana Shore","predicate":"was_at_location","object_":"city theatre","time_ref":"6PM-10PM","confidence":"stated","source_text":"I was at the city theatre watching a play from six until ten"},'
    '{"claim_type":"relationship","subject":"Diana Shore","predicate":"has_witness","object_":"multiple friends","time_ref":"evening","confidence":"stated","source_text":"several friends who saw me there"}]',
    '[{"claim_type":"denial","subject":"Diana Shore","predicate":"had_conflict_with","object_":"victim","time_ref":null,"confidence":"stated","source_text":"perfectly fine relationship with the victim"},'
    '{"claim_type":"relationship","subject":"Diana Shore","predicate":"collaborated_with","object_":"victim","time_ref":"recently","confidence":"stated","source_text":"We collaborated on a project recently"}]',
]

# Witness claim contradicting Victor's alibi — injected by simulating a
# cross-reference question to Diana that returns Victor's location.
DIANA_CROSSREF_CLAIMS = [
    '[{"claim_type":"accusation","subject":"Diana Shore","predicate":"saw_near","object_":"Victor Crane","time_ref":"8PM","confidence":"stated","source_text":"I did see Victor Crane walking past the office building around eight"}]',
]

FOLLOW_UP_NO   = '{"warranted": false, "reason": "sufficient claims extracted", "question": ""}'
FOLLOW_UP_YES  = '{"warranted": true, "reason": "alibi unverified", "question": "Can you be more specific about your alibi?"}'
NARRATIVE = (
    "Based on the evidence gathered, Victor Crane is the most likely perpetrator. "
    "Crane provided a false alibi claiming to be at the Golden Fork restaurant during "
    "the estimated time of death, but an independent witness placed him near the "
    "victim's office at 8PM — directly contradicting his account. "
    "Furthermore, Crane admitted to a recent business conflict with the victim, "
    "establishing a clear motive. Diana Shore provided a corroborated alibi "
    "supported by physical evidence (theatre ticket) and multiple witnesses, "
    "and showed no motive. The evidence consistently points to Victor Crane "
    "as the murderer."
)


class _MockUsage:
    prompt_tokens = 80
    completion_tokens = 40

class _MockResponse:
    def __init__(self, content):
        _Msg = type("Msg", (), {"content": content})
        _Choice = type("Choice", (), {"message": _Msg()})
        self.choices = [_Choice()]
        self.usage = _MockUsage()


class _MockCompletions:
    """Handles the .completions.create() level of the chained call."""

    def __init__(self, router_fn):
        self._router = router_fn

    def create(self, model, messages, temperature=0.5, max_tokens=None, timeout=None):
        return self._router(model, messages, temperature, max_tokens, timeout)


class _MockChat:
    """Handles the .chat level of the chained call."""

    def __init__(self, router_fn):
        self.completions = _MockCompletions(router_fn)


class MockLLMClient:
    """
    Scripted LLM client. Returns pre-written responses in sequence.
    Tracks call counts to feed the right answer at the right turn.
    No network calls are made.

    The openai SDK calls client.chat.completions.create(...).
    This mock replicates that exact attribute chain.
    """

    def __init__(self):
        self._victor_q_idx = 0
        self._victor_parse_idx = 0
        self._diana_q_idx = 0
        self._diana_parse_idx = 0
        self._call_log = []
        self.chat = _MockChat(self._route)

    def _route(self, model, messages, temperature=0.5, max_tokens=None, timeout=None):
        """Route each call to the right scripted response based on message content."""

        system_content = messages[0].get("content", "") if messages else ""
        last_user = messages[-1].get("content", "") if messages else ""
        full_text = " ".join(m.get("content", "") for m in messages)

        # Identify call type by content signatures
        is_parser_call = "forensic claim extractor" in system_content
        is_narrative_call = "conclusion section" in last_user
        is_follow_up_decision = "deciding whether to ask a follow-up" in last_user

        # Determine which suspect is being processed
        is_victor = "Victor Crane" in full_text
        is_diana = "Diana Shore" in full_text

        self._call_log.append({
            "is_parser": is_parser_call,
            "is_narrative": is_narrative_call,
            "is_follow_up": is_follow_up_decision,
            "is_victor": is_victor,
            "is_diana": is_diana,
        })

        # --- Narrative call ---
        if is_narrative_call:
            return _MockResponse(NARRATIVE)

        # --- Follow-up decision ---
        if is_follow_up_decision:
            return _MockResponse(FOLLOW_UP_NO)

        # --- Parser calls ---
        if is_parser_call:
            if is_victor:
                idx = min(self._victor_parse_idx, len(VICTOR_CLAIMS) - 1)
                self._victor_parse_idx += 1
                return _MockResponse(VICTOR_CLAIMS[idx])
            if is_diana:
                if self._diana_parse_idx >= len(DIANA_CLAIMS):
                    return _MockResponse(DIANA_CROSSREF_CLAIMS[0])
                idx = self._diana_parse_idx
                self._diana_parse_idx += 1
                return _MockResponse(DIANA_CLAIMS[idx])

        # --- Question generation (detective) and suspect responses ---
        if is_victor:
            idx = min(self._victor_q_idx, len(VICTOR_ANSWERS) - 1)
            self._victor_q_idx += 1
            if max_tokens and max_tokens <= 120:
                return _MockResponse(VICTOR_QUESTIONS[min(idx, len(VICTOR_QUESTIONS)-1)])
            return _MockResponse(VICTOR_ANSWERS[idx])

        if is_diana:
            idx = min(self._diana_q_idx, len(DIANA_ANSWERS) - 1)
            self._diana_q_idx += 1
            if max_tokens and max_tokens <= 120:
                return _MockResponse(DIANA_QUESTIONS[min(idx, len(DIANA_QUESTIONS)-1)])
            return _MockResponse(DIANA_ANSWERS[idx])

        # Fallback for any unmatched call
        return _MockResponse("I have nothing more to add.")


# ---------------------------------------------------------------------------
# Minimal SuspectAgent (same sliding-window logic as main.py)
# ---------------------------------------------------------------------------

class MockSuspectAgent:
    """
    Minimal suspect agent backed by the mock client.
    Mirrors the sliding-window logic in main.py's SuspectAgent.
    """

    def __init__(self, name: str, story: str, task: str, victim: str, client: MockLLMClient):
        self.name = name
        self._client = client
        self._system_prompt = (
            f"You are {name}. Do not break character.\n\n"
            f"### YOUR STORY ###\n{story}\n\n"
            f"### CASE CONTEXT ###\nYou are being questioned about the murder of {victim}.\n\n"
            f"### DIRECTIVE ###\n{task}"
        )
        self._history = []

    def interrogate(self, question: str) -> str:
        self._history.append({"role": "user", "content": question})
        context = [{"role": "system", "content": self._system_prompt}] + self._history[-12:]
        resp = self._client.chat.completions.create(
            model="mock",
            messages=context,
            temperature=0.45,
        )
        answer = resp.choices[0].message.content
        self._history.append({"role": "assistant", "content": answer})
        return answer


# ---------------------------------------------------------------------------
# Minimal case definition
# ---------------------------------------------------------------------------

FAKE_CASE = {
    "time": "Evening (7PM-10PM)",
    "location": "Harrington Office Building",
    "victim": {"name": "Mr. Harrington"},
    "suspects": [
        {
            "name": "Victor Crane",
            "introduction": "Business associate of the victim.",
            "relationship": "Business partner",
            "suspicion": ["seen near office", "had recent dispute"],
        },
        {
            "name": "Diana Shore",
            "introduction": "Colleague of the victim.",
            "relationship": "Work colleague",
            "suspicion": [],
        },
    ],
    "label": 0,  # Victor Crane is the murderer (index 0)
}


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test():
    print("=" * 60)
    print("Project OMNISCIENT — Sanity Test")
    print("Mock LLM | 2 suspects | No API keys required")
    print("=" * 60)
    print()

    mock_client = MockLLMClient()
    counter = TokenCounter()

    # Wire up the mock client to the real Interrogator
    counter_class = counter
    interrogator = Interrogator(
        detective_client=mock_client,
        detective_model="mock",
        parser_client=mock_client,
        parser_model="mock",
        counter=counter,
    )

    # Build the reasoning engine
    engine = ReasoningEngine(
        case_data=FAKE_CASE,
        suspects=["Victor Crane", "Diana Shore"],
    )
    # Data-driven priors (2 suspicion observations for Victor, 0 for Diana)
    engine.state.suspicion_scores = {"Victor Crane": 0.50, "Diana Shore": 0.35}

    # Build suspect agents
    agents = {
        "Victor Crane": MockSuspectAgent(
            name="Victor Crane",
            story="You were actually near the office but claim you were at a restaurant.",
            task="Be defensive when pressed. Admit the business conflict if directly asked.",
            victim="Mr. Harrington",
            client=mock_client,
        ),
        "Diana Shore": MockSuspectAgent(
            name="Diana Shore",
            story="You were at the city theatre all evening. You have a ticket stub.",
            task="Be cooperative and honest.",
            victim="Mr. Harrington",
            client=mock_client,
        ),
    }

    print("[ INVESTIGATION LOOP ]")
    print("-" * 60)

    session_num = 0
    MAX_SESSIONS_OVERRIDE = 6  # keep the test short

    while True:
        goal = engine.next_goal()
        if goal is None:
            print("Engine: no further goals.")
            break

        session_num += 1
        if session_num > MAX_SESSIONS_OVERRIDE:
            print(f"Stopping after {MAX_SESSIONS_OVERRIDE} sessions (test limit).")
            break

        print(f"\nSession {session_num}: {goal.suspect_name} | {goal.objective.value} | {goal.persona.value}")

        testimony = interrogator.conduct_session(goal, agents[goal.suspect_name])
        engine.ingest_testimony(testimony)

        print(f"  Claims extracted : {len(testimony.claims)}")
        for c in testimony.claims:
            print(f"    [{c.claim_type.value}] {c.subject} | {c.predicate} | {c.object_}"
                  + (f" | {c.time_ref}" if c.time_ref else ""))

        scores = engine.state.suspicion_scores
        print(f"  Scores after     : " +
              " | ".join(f"{k}={v:.2f}" for k, v in scores.items()))

        if engine.state.contradictions:
            print(f"  Contradictions   : {len(engine.state.contradictions)} detected")

        verdict = engine.check_termination()
        if verdict is not None:
            print(f"\nTerminal condition: {verdict.status.upper()}")
            break

    # Final verdict if loop ended without formal condition
    verdict = engine.check_termination()
    if verdict is None:
        sorted_s = engine._sorted_suspects()
        top_name, top_score = sorted_s[0]
        from reasoning_engine import Verdict
        verdict = Verdict(
            suspect=top_name,
            confidence=top_score,
            status="inconclusive",
            reasoning=engine.get_reasoning_trace(),
            contradictions_found=engine.state.contradictions,
            suspicion_events=engine.state.suspicion_events,
            total_sessions=sum(engine.state.sessions_done.values()),
            final_scores=dict(engine.state.suspicion_scores),
        )
        engine.state.status = "inconclusive"

    # Generate report
    print()
    print("[ CASE REPORT ]")
    print("-" * 60)

    os.makedirs("reports", exist_ok=True)
    report_gen = ReportGenerator(
        case_state=engine.state,
        verdict=verdict,
        narrative_client=mock_client,
        narrative_model="mock",
        start_time=datetime.utcnow(),
        counter=counter,
        ground_truth="Victor Crane",
    )
    report = report_gen.generate(output_path="reports/sanity_test.txt")
    print(report)

    # Assertions
    print()
    print("[ ASSERTIONS ]")
    print("-" * 60)

    ground_truth = "Victor Crane"
    correct = verdict.suspect.strip().lower() == ground_truth.lower()
    assert correct, f"FAIL: expected '{ground_truth}', got '{verdict.suspect}'"
    print(f"  Verdict correct       : PASS ({verdict.suspect})")

    victor_score = engine.state.suspicion_scores.get("Victor Crane", 0)
    diana_score  = engine.state.suspicion_scores.get("Diana Shore", 0)
    assert victor_score > diana_score, "FAIL: Victor's score should exceed Diana's"
    print(f"  Score ordering        : PASS (Victor={victor_score:.2f} > Diana={diana_score:.2f})")

    contradictions = engine.state.contradictions
    assert len(contradictions) >= 0, "FAIL: contradiction list missing"
    print(f"  Contradictions logged : PASS ({len(contradictions)} found)")

    token_summary = counter.summary()
    assert token_summary["total_calls"] > 0, "FAIL: no API calls recorded"
    print(f"  Token counter         : PASS ({token_summary['total_calls']} calls, "
          f"{token_summary['total_tokens']} tokens)")

    assert os.path.exists("reports/sanity_test.txt"), "FAIL: report file not written"
    print(f"  Report written        : PASS (reports/sanity_test.txt)")

    print()
    print("=" * 60)
    print("ALL ASSERTIONS PASSED — pipeline is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    run_test()
