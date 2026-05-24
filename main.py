"""
Project OMNISCIENT — Entry Point.

This script is the starting gate for the entire application.
It handles:
  1. Logging initialisation (first, before anything else).
  2. Dataset loading and case selection via CLI argument.
  3. LLM client initialisation:
       - All investigation roles (suspects, detective, parser) -> local Ollama.
         Running locally means no token-limit costs and no rate limits,
         making trial-and-error testing completely free and unrestricted.
       - Narrative report -> Gemini if GEMINI_API_KEY is set, else Ollama.
         This is one call at the end (~400 tokens) and is optional.
  4. Suspect agent initialisation with data-driven priors.
  5. Investigation loop.
  6. Case report generation and accuracy reporting.

Persona stability strategy (no cloud LLM required):
  - The suspect system prompt is embedded once at session construction and
    always sits at position [0] of the message list, anchoring the model.
  - A sliding window keeps only the last SUSPECT_HISTORY_TURNS Q&A pairs
    in the context, preventing old exchanges from diluting the system prompt.
  - Temperature 0.45 for suspects produces consistent in-character responses
    without being entirely deterministic.

Usage:
    python main.py            # runs case index 0
    python main.py 42         # runs case index 42

Environment variables (see .env.example):
    OLLAMA_HOST         Optional. Default: http://localhost:11434
    DETECTIVE_MODEL     Optional. Default: qwen2.5:7b
    GEMINI_API_KEY      Optional. Only needed for the narrative LLM call.
    GEMINI_MODEL        Optional. Default: models/gemini-2.5-flash-preview-05-20
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from interrogator import Interrogator
from reasoning_engine import ReasoningEngine, Verdict
from report_generator import ReportGenerator
from token_counter import TokenCounter

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Mapping from suspicion-observation count to prior suspicion score.
# The dataset 'suspicion' field is a list of observation strings; length is
# used as a proxy for prior strength (more observations = stronger prior).
SUSPICION_PRIOR_BY_COUNT: dict = {
    0: 0.35,
    1: 0.40,
    2: 0.50,
    3: 0.60,
    4: 0.65,
    5: 0.70,
}
SUSPICION_PRIOR_MAX: float = 0.70

# Output directories
REPORTS_DIR: str = "reports"
LOGS_DIR: str = "logs"

# Suspect conversation history window.
# Only the last N complete Q&A pairs (plus the system prompt) are sent to
# the model. This prevents old context from diluting the persona anchored
# in the system prompt, achieving stability without needing a massive
# context window or a cloud LLM.
SUSPECT_HISTORY_TURNS: int = 6

# Temperatures.
# Suspects use 0.45: enough creative variation to feel natural but
# constrained enough that the system prompt persona dominates.
# The detective uses lower temperatures for focused, targeted questions.
SUSPECT_TEMPERATURE: float = 0.45


# ---------------------------------------------------------------------------
# SuspectAgent
# ---------------------------------------------------------------------------

class SuspectAgent:
    """
    Represents a single suspect in the investigation.

    Wraps a local Ollama client pre-loaded with the suspect's story,
    timeline, motive, and behavioural directive.

    Persona stability is achieved through three mechanisms (no cloud LLM needed):
      1. The system prompt is always at index [0] of every request, anchoring
         the character no matter how long the session becomes.
      2. A sliding window keeps only the last SUSPECT_HISTORY_TURNS Q&A pairs
         in the context. Old exchanges that might dilute the persona are dropped.
      3. Temperature 0.45 keeps responses consistent without being robotic.

    Dataset fields used by the DETECTIVE SYSTEM (restricted per specification):
        name, introduction, relationship, suspicion.

    Additional fields loaded here for the SUSPECT AGENT'S own persona
    (not exposed to the detective logic):
        story, task.

    Args:
        suspect_data   : Suspect dict from the dataset.
        victim_name    : Name of the victim (context for the suspect).
        model_client   : OpenAI-compatible client (Ollama or Gemini).
        model_name     : Model identifier string.
        max_turns      : Maximum Q&A pairs to keep in the sliding window.
    """

    def __init__(
        self,
        suspect_data: dict,
        victim_name: str,
        model_client: OpenAI,
        model_name: str,
        max_turns: int = SUSPECT_HISTORY_TURNS,
    ) -> None:
        self.client = model_client
        self.model_name = model_name
        self.name = suspect_data.get("name", "Unknown Suspect")
        self._max_turns = max_turns

        self._system_prompt = (
            f"You are {self.name}. Do not break character under any circumstances.\n\n"
            f"### YOUR BACKGROUND ###\n{suspect_data.get('introduction', '')}\n\n"
            f"### YOUR STORY AND TIMELINE ###\n{suspect_data.get('story', '')}\n\n"
            f"### CASE CONTEXT ###\n"
            f"A detective is interrogating you regarding the murder of {victim_name}.\n\n"
            f"### YOUR BEHAVIOURAL DIRECTIVE ###\n"
            f"{suspect_data.get('task', 'Answer questions as your character would.')}\n\n"
            "Maintain your character's personality, knowledge, and secrets consistently. "
            "Do not contradict your own prior statements within this interrogation."
        )

        # Full history stored here; only a window of it is sent to the model.
        self._full_history: list = []

    def _build_context(self) -> list:
        """
        Build the message list to send to the model.

        Always starts with the system prompt. Then appends at most
        max_turns complete Q&A pairs (2 messages each) from the tail of
        the full history. This caps the input tokens per call regardless
        of how many sessions have occurred.
        """
        system_msg = {"role": "system", "content": self._system_prompt}
        # Each turn = 2 messages (user question + assistant answer)
        window_messages = self._full_history[-(self._max_turns * 2):]
        return [system_msg] + window_messages

    def interrogate(self, question: str) -> str:
        """
        Respond to one detective question in character.

        The full exchange is appended to _full_history for coherence;
        only a sliding window of that history is sent to the model.

        Args:
            question : The detective's question as a plain string.

        Returns:
            The suspect's response as a plain string.
        """
        self._full_history.append({"role": "user", "content": question})

        context = self._build_context()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=context,
            temperature=SUSPECT_TEMPERATURE,
        )

        answer = response.choices[0].message.content
        self._full_history.append({"role": "assistant", "content": answer})
        return answer


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

def _build_ollama_client(host: str) -> OpenAI:
    """
    Build an OpenAI-compatible client pointing at a local Ollama server.

    The openai SDK treats this identically to any other endpoint;
    'ollama' is passed as the api_key only because the SDK requires the
    field — Ollama does not validate it.
    """
    return OpenAI(
        base_url=f"{host}/v1",
        api_key="ollama",
    )


def _build_gemini_client(api_key: str) -> OpenAI:
    """Build an OpenAI-compatible client pointing at the Gemini API."""
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )


def _ollama_is_reachable(host: str, model: str) -> bool:
    """
    Check whether Ollama is running and the requested model is pulled.

    Sends a minimal 1-token completion with a 5-second timeout.
    Returns True on success, False on any connection or model error.
    """
    try:
        client = _build_ollama_client(host)
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            timeout=5,
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Prior computation
# ---------------------------------------------------------------------------

def compute_priors(suspects: list) -> dict:
    """
    Derive initial suspicion priors from the dataset 'suspicion' field.

    Args:
        suspects : List of suspect dicts from the dataset.

    Returns:
        Dict mapping suspect name -> float prior in [0.35, 0.70].
    """
    priors = {}
    for suspect in suspects:
        name = suspect.get("name", "Unknown")
        suspicion_obs = suspect.get("suspicion", [])
        count = len(suspicion_obs) if isinstance(suspicion_obs, list) else 0
        prior = SUSPICION_PRIOR_BY_COUNT.get(count, SUSPICION_PRIOR_MAX)
        priors[name] = prior
        logging.info(
            "Suspicion prior | suspect=%s | observations=%d | prior=%.2f",
            name, count, prior,
        )
    return priors


# ---------------------------------------------------------------------------
# Investigation loop
# ---------------------------------------------------------------------------

def run_investigation(
    suspect_agents: dict,
    interrogator: Interrogator,
    engine: ReasoningEngine,
    session_delay: int = 0,
) -> None:
    """
    Main investigation loop: request goals from the engine, conduct sessions,
    ingest testimony, and check for a terminal verdict after each session.

    The loop exits when:
      - engine.next_goal() returns None (budget exhausted or no valid goals remain).
      - engine.check_termination() returns a Verdict (formal condition met).

    Args:
        suspect_agents : Dict mapping suspect name -> SuspectAgent.
        interrogator   : Configured Interrogator instance.
        engine         : ReasoningEngine with initialised priors.
        session_delay  : Seconds to sleep between sessions (for rate limiting).
    """
    logging.info(
        "Investigation START | case_id=%s | victim=%s | suspects=%s",
        engine.state.case_id,
        engine.state.victim_name,
        engine.state.suspects,
    )

    session_number = 0

    while True:
        goal = engine.next_goal()
        if goal is None:
            logging.info("No further goals from engine. Ending investigation loop.")
            break

        session_number += 1
        logging.info(
            "Session %d | suspect=%s | objective=%s | persona=%s",
            session_number,
            goal.suspect_name,
            goal.objective.value,
            goal.persona.value,
        )

        suspect_agent = suspect_agents.get(goal.suspect_name)
        if suspect_agent is None:
            logging.error(
                "No SuspectAgent found for name=%s. Skipping.", goal.suspect_name
            )
            continue

        try:
            testimony = interrogator.conduct_session(goal, suspect_agent)
        except Exception as exc:
            logging.error(
                "Session failed | suspect=%s | error=%s",
                goal.suspect_name, exc, exc_info=True,
            )
            continue

        engine.ingest_testimony(testimony)

        if session_delay > 0:
            logging.debug("Rate-limit pause: %ds", session_delay)
            time.sleep(session_delay)

        verdict = engine.check_termination()
        if verdict is not None:
            logging.info(
                "Terminal condition met | status=%s | suspect=%s | confidence=%.2f",
                verdict.status, verdict.suspect, verdict.confidence,
            )
            break

    logging.info(
        "Investigation END | sessions=%d | status=%s",
        session_number, engine.state.status,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate one complete investigation run for a single dataset case.
    """

    # ------------------------------------------------------------------
    # 1. Parse CLI arguments
    # ------------------------------------------------------------------
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    case_id = f"{index:03d}"

    # ------------------------------------------------------------------
    # 2. Initialise logging (before everything else)
    # ------------------------------------------------------------------
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"case_{case_id}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    logging.info("Logging initialised | log_file=%s", log_path)

    # ------------------------------------------------------------------
    # 3. Load environment variables
    # ------------------------------------------------------------------
    load_dotenv()

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    local_model = os.environ.get("DETECTIVE_MODEL", "qwen2.5:7b")

    # Gemini is optional — used only for the narrative section of the report
    # (one call, ~400 tokens). If the key is absent, the local model is used.
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    gemini_model = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash-preview-05-20")

    # ------------------------------------------------------------------
    # 4. Initialise local Ollama client (used for all investigation roles)
    #
    # All suspect agents, the detective, and the claim parser run through
    # the same local Ollama endpoint. This eliminates token-limit costs,
    # API rate limits, and the need for any cloud API key during the
    # investigation itself.
    # Persona stability is achieved via sliding-window context management
    # (SuspectAgent._build_context) and temperature settings — not by
    # relying on a large cloud context window.
    # ------------------------------------------------------------------
    logging.info(
        "Checking local Ollama at %s with model '%s'...", ollama_host, local_model
    )
    if not _ollama_is_reachable(ollama_host, local_model):
        logging.critical(
            "Ollama is not reachable at %s or model '%s' is not pulled. "
            "Run 'bash setup.sh' to install Ollama and pull the model, then retry.",
            ollama_host, local_model,
        )
        sys.exit(1)

    local_client = _build_ollama_client(ollama_host)
    logging.info("Local Ollama client ready | host=%s | model=%s", ollama_host, local_model)

    # Narrative client: prefer Gemini (better at coherent narrative writing)
    # but fall back to local Ollama if no Gemini key is provided.
    if gemini_api_key:
        narrative_client = _build_gemini_client(gemini_api_key)
        narrative_model = gemini_model
        logging.info("Narrative client: Gemini | model=%s", gemini_model)
    else:
        narrative_client = local_client
        narrative_model = local_model
        logging.info(
            "GEMINI_API_KEY not set. Narrative will use local model '%s'.", local_model
        )

    # ------------------------------------------------------------------
    # 5. Load dataset and select case
    # ------------------------------------------------------------------
    dataset_path = "dataset.json"
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        logging.info("Dataset loaded | path=%s | total_cases=%d", dataset_path, len(dataset))
    except FileNotFoundError:
        logging.critical("Dataset not found at '%s'.", dataset_path)
        sys.exit(1)

    if index < 0 or index >= len(dataset):
        logging.critical(
            "Case index %d is out of range. Dataset has %d cases.", index, len(dataset)
        )
        sys.exit(1)

    case = dataset[index]
    victim_name = case.get("victim", {}).get("name", "Unknown")

    # Ground truth (murderer name) is used only for accuracy evaluation
    label_index = case.get("label")
    ground_truth = None
    if label_index is not None:
        suspect_list = case.get("suspects", [])
        if 0 <= label_index < len(suspect_list):
            ground_truth = suspect_list[label_index].get("name")

    logging.info(
        "Case loaded | index=%d | victim=%s | suspects=%d | ground_truth=%s",
        index, victim_name, len(case.get("suspects", [])),
        ground_truth or "not available",
    )

    # ------------------------------------------------------------------
    # 6. Initialise suspect agents using the local Ollama client.
    #    Only the allowed fields are exposed to the detective system:
    #    name, introduction, relationship, suspicion.
    #    The suspect agent itself receives the full story and task for
    #    its own persona — these are never read by the detective logic.
    # ------------------------------------------------------------------
    suspect_agents = {}
    for suspect_data in case.get("suspects", []):
        name = suspect_data.get("name", "Unknown")
        suspect_agents[name] = SuspectAgent(
            suspect_data=suspect_data,
            victim_name=victim_name,
            model_client=local_client,
            model_name=local_model,
        )
        logging.info("SuspectAgent created | name=%s | model=%s", name, local_model)

    if not suspect_agents:
        logging.critical("No suspect agents could be created. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 7. Initialise the reasoning engine with data-driven priors
    # ------------------------------------------------------------------
    engine = ReasoningEngine(
        case_data=case,
        suspects=list(suspect_agents.keys()),
    )
    priors = compute_priors(case.get("suspects", []))
    engine.state.suspicion_scores = priors
    logging.info("Data-driven priors applied | %s", priors)

    # ------------------------------------------------------------------
    # 8. Initialise shared token counter and interrogator.
    #    Both the detective (question generation) and the parser (claim
    #    extraction) use the same local Ollama client and model.
    # ------------------------------------------------------------------
    counter = TokenCounter()

    interrogator = Interrogator(
        detective_client=local_client,
        detective_model=local_model,
        parser_client=local_client,
        parser_model=local_model,
        counter=counter,
    )

    # ------------------------------------------------------------------
    # 9. Run the investigation.
    #    No session delay needed — local inference has no rate limits.
    # ------------------------------------------------------------------
    start_time = datetime.utcnow()
    logging.info("Investigation starting | start=%s", start_time.isoformat())

    try:
        run_investigation(suspect_agents, interrogator, engine, session_delay=0)
    except Exception as exc:
        logging.critical(
            "Unhandled exception during investigation: %s", exc, exc_info=True
        )

    # ------------------------------------------------------------------
    # 10. Collect final verdict
    #     check_termination() may return None if the loop ended because
    #     next_goal() ran out of goals rather than a formal condition.
    #     In that case, build a fallback inconclusive verdict.
    # ------------------------------------------------------------------
    verdict = engine.check_termination()
    if verdict is None:
        sorted_s = engine._sorted_suspects()
        top_suspect = sorted_s[0][0] if sorted_s else ""
        top_score = sorted_s[0][1] if sorted_s else 0.0
        verdict = Verdict(
            suspect=top_suspect,
            confidence=top_score,
            status="inconclusive",
            reasoning=engine.get_reasoning_trace(),
            contradictions_found=engine.state.contradictions,
            suspicion_events=engine.state.suspicion_events,
            total_sessions=sum(engine.state.sessions_done.values()),
            final_scores=dict(engine.state.suspicion_scores),
        )
        engine.state.status = "inconclusive"
        logging.warning("No formal terminal verdict. Defaulting to inconclusive.")

    # ------------------------------------------------------------------
    # 11. Generate and write the case report
    # ------------------------------------------------------------------
    report_path = os.path.join(REPORTS_DIR, f"case_{case_id}.txt")

    report_generator = ReportGenerator(
        case_state=engine.state,
        verdict=verdict,
        narrative_client=narrative_client,  # Gemini if key set, else local
        narrative_model=narrative_model,
        start_time=start_time,
        counter=counter,
        ground_truth=ground_truth,
    )

    try:
        report_text = report_generator.generate(output_path=report_path)
        print("\n" + "=" * 80)
        print(report_text)
        print("=" * 80)
        logging.info("Case complete | report=%s", report_path)
    except Exception as exc:
        logging.error("Report generation failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # 12. Print summary to stdout
    # ------------------------------------------------------------------
    if ground_truth:
        correct = verdict.suspect.strip().lower() == ground_truth.strip().lower()
        result = "CORRECT" if correct else "INCORRECT"
        print(f"\nAccuracy: {result} | Answer: {verdict.suspect} | Truth: {ground_truth}")

    token_summary = counter.summary()
    print(
        f"Tokens used: {token_summary['total_tokens']} "
        f"(prompt={token_summary['prompt_tokens']}, "
        f"completion={token_summary['completion_tokens']}, "
        f"calls={token_summary['total_calls']})"
    )


if __name__ == "__main__":
    main()
