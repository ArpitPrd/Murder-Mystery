"""
Case Report Generator — Part 3 of Project OMNISCIENT.

Produces a comprehensive, human-readable case report for every investigation.
The report covers:
  1. Case overview (victim, time, location)
  2. Verdict with final suspicion scores
  3. Suspect summary table
  4. Chronological reconstruction (one LLM call)
  5. Full investigation timeline (all sessions, turns, claims)
  6. Reasoning trace (every suspicion score event in order)
  7. Contradictions detected and whether resolved
  8. Errors and anomalies (parse failures, zero-claim sessions)
  9. Computational cost (sessions, turns, tokens, elapsed time)

Design:
  - Sections 1–3 and 5–9 are assembled deterministically from the CaseState
    and Verdict data structures — no LLM calls required.
  - Section 4 (Chronological Reconstruction) makes exactly one LLM call,
    capped at 400 completion tokens. If the call fails, a deterministic
    fallback is used and the failure is logged.
  - Token counts from the full investigation are reported in Section 9
    via the shared TokenCounter.
"""

import logging
import os
from datetime import datetime
from openai import OpenAI

from interrogator import Testimony, Claim
from reasoning_engine import CaseState, Verdict, Contradiction, SuspicionEvent
from token_counter import TokenCounter

logger = logging.getLogger(__name__)

# Width of section divider lines in the report
REPORT_WIDTH = 80


class ReportGenerator:
    """
    Assembles and writes the final case report.

    Args:
        case_state       : Completed CaseState from the ReasoningEngine.
        verdict          : Verdict produced by ReasoningEngine.check_termination().
        narrative_client : OpenAI-compatible client for the reconstruction narrative.
        narrative_model  : Model identifier for the narrative call.
        start_time       : datetime when the investigation began (for elapsed time).
        counter          : Shared TokenCounter recording all API usage.
        ground_truth     : Name of the actual murderer (for validation accuracy).
                           Pass None when unknown.
    """

    def __init__(
        self,
        case_state: CaseState,
        verdict: Verdict,
        narrative_client: OpenAI,
        narrative_model: str,
        start_time: datetime,
        counter: TokenCounter,
        ground_truth: str = None,
    ) -> None:
        self.case_state = case_state
        self.verdict = verdict
        self.narrative_client = narrative_client
        self.narrative_model = narrative_model
        self.start_time = start_time
        self.counter = counter
        self.ground_truth = ground_truth

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _header(self, title: str, char: str = "-") -> str:
        """Return a section header with a full-width divider below it."""
        return f"{title}\n{char * REPORT_WIDTH}"

    def _divider(self, char: str = "-") -> str:
        """Return a plain full-width divider line."""
        return char * REPORT_WIDTH

    def _score_bar(self, score: float, width: int = 20) -> str:
        """
        Render a suspicion score as an ASCII progress bar.
        Example: 0.75 -> [###############     ]
        """
        filled = int(score * width)
        empty = width - filled
        return f"[{'#' * filled}{' ' * empty}]"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(self, output_path: str) -> str:
        """
        Assemble the complete case report, write it to output_path, and return it.

        Args:
            output_path : Destination file path (e.g. reports/case_001.txt).

        Returns:
            The complete report as a plain-text string.
        """
        logger.info(
            "Generating case report | case_id=%s | output=%s",
            self.case_state.case_id,
            output_path,
        )

        sections = [
            self._section_overview(),
            self._section_verdict(),
            self._section_suspect_table(),
            self._section_reconstruction(),
            self._section_investigation_timeline(),
            self._section_reasoning_trace(),
            self._section_contradictions(),
            self._section_errors(),
            self._section_computational_cost(),
        ]

        report = "\n\n".join(sections)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info("Report written | path=%s", output_path)
        except OSError as exc:
            logger.error("Failed to write report | path=%s | error=%s", output_path, exc)

        return report

    # ------------------------------------------------------------------
    # Report sections
    # ------------------------------------------------------------------

    def _section_overview(self) -> str:
        """
        Section 1 — Case overview.
        Static metadata: victim, time, location, case ID, suspects listed.
        """
        lines = [
            self._header("CASE REPORT — PROJECT OMNISCIENT", char="="),
            f"Case ID       : {self.case_state.case_id}",
            f"Generated     : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            self._divider("="),
            self._header("1. CASE OVERVIEW"),
            f"Victim        : {self.case_state.victim_name}",
            f"Time of Crime : {self.case_state.crime_time}",
            f"Location      : {self.case_state.crime_location}",
            f"Suspects      : {', '.join(self.case_state.suspects)}",
        ]
        return "\n".join(lines)

    def _section_verdict(self) -> str:
        """
        Section 2 — Verdict.
        Named murderer (or unsolvable/inconclusive), confidence, ranked scores,
        and accuracy check if ground truth is available.
        """
        lines = [self._header("2. VERDICT")]

        status = self.verdict.status.upper()
        lines.append(f"Status        : {status}")

        if self.verdict.status == "solved":
            lines.append(f"Identified    : {self.verdict.suspect}")
            lines.append(f"Confidence    : {self.verdict.confidence:.1%}")
        elif self.verdict.status == "inconclusive":
            lines.append(f"Most Likely   : {self.verdict.suspect} (inconclusive)")
            lines.append(f"Confidence    : {self.verdict.confidence:.1%}")
            lines.append(
                "NOTE: The investigation could not separate the top suspects by "
                "the required margin. See the reasoning trace for details."
            )
        else:
            lines.append(
                "The investigation exhausted its session budget without reaching a "
                "confident verdict. See the errors section for details."
            )

        # Accuracy check against ground truth (for validation runs)
        if self.ground_truth:
            correct = (
                self.verdict.suspect.strip().lower() == self.ground_truth.strip().lower()
            )
            result_str = "CORRECT" if correct else "INCORRECT"
            lines.append("")
            lines.append(f"Accuracy Check: {result_str}")
            lines.append(f"  Ground Truth: {self.ground_truth}")
            lines.append(f"  System Answer: {self.verdict.suspect or 'None'}")

        lines.append("")
        lines.append("Final Suspicion Scores (all suspects):")
        for suspect, score in sorted(
            self.verdict.final_scores.items(), key=lambda x: x[1], reverse=True
        ):
            bar = self._score_bar(score)
            lines.append(f"  {suspect:<30} {score:.2f}  {bar}")

        return "\n".join(lines)

    def _section_suspect_table(self) -> str:
        """
        Section 3 — Suspect summary table.
        One row per suspect: sessions completed, claims extracted, evasiveness.
        """
        lines = [self._header("3. SUSPECT SUMMARY")]

        col = "{:<30} {:>8} {:>9} {:>8} {:>13}"
        lines.append(col.format("Suspect", "Score", "Sessions", "Claims", "Max Evasive"))
        lines.append(self._divider("-"))

        for suspect in self.case_state.suspects:
            score = self.case_state.suspicion_scores.get(suspect, 0.0)
            sessions = self.case_state.sessions_done.get(suspect, 0)
            suspect_testimonies = [
                t for t in self.case_state.testimonies if t.suspect_name == suspect
            ]
            total_claims = sum(len(t.claims) for t in suspect_testimonies)
            max_evasiveness = max(
                (t.evasiveness_score for t in suspect_testimonies), default=0
            )
            lines.append(
                col.format(suspect, f"{score:.2f}", sessions, total_claims, max_evasiveness)
            )

        return "\n".join(lines)

    def _section_reconstruction(self) -> str:
        """
        Section 4 — Chronological reconstruction.
        Exactly one LLM call to produce a 2-3 paragraph narrative of what
        likely happened, grounded in the verdict and evidence.
        Falls back to a deterministic summary if the call fails.
        """
        lines = [self._header("4. CHRONOLOGICAL RECONSTRUCTION")]
        narrative = self._reconstruction_narrative()
        lines.append(narrative)
        return "\n".join(lines)

    def _section_investigation_timeline(self) -> str:
        """
        Section 5 — Full investigation timeline.
        Every session in chronological order: suspect, objective, persona,
        full Q&A transcript, and extracted claims per turn.
        """
        lines = [self._header("5. INVESTIGATION TIMELINE")]

        if not self.case_state.testimonies:
            lines.append("No sessions were conducted.")
            return "\n".join(lines)

        for idx, testimony in enumerate(self.case_state.testimonies, start=1):
            lines.append(
                f"\nSession {idx} | Suspect: {testimony.suspect_name} | "
                f"Objective: {testimony.objective.value} | "
                f"Persona: {testimony.persona_used.value} | "
                f"ID: {testimony.testimony_id}"
            )
            lines.append(self._divider("-"))

            for turn in testimony.turns:
                lines.append(f"  Turn {turn['turn']}")
                lines.append(f"  Detective : {turn['question']}")
                lines.append(f"  Suspect   : {turn['raw_answer']}")
                lines.append(f"  Claims extracted: {turn['claims_extracted']}")
                lines.append("")

            if testimony.claims:
                lines.append("  Structured Claims:")
                for claim in testimony.claims:
                    time_str = f" | time: {claim.time_ref}" if claim.time_ref else ""
                    lines.append(
                        f"    [{claim.claim_type.value}] "
                        f"{claim.subject} | {claim.predicate} | {claim.object_}"
                        f"{time_str} | confidence: {claim.confidence}"
                    )

            lines.append(
                f"\n  Session Assessment | tone: {testimony.emotional_tone} | "
                f"evasiveness: {testimony.evasiveness_score}/5"
            )

        return "\n".join(lines)

    def _section_reasoning_trace(self) -> str:
        """
        Section 6 — Reasoning trace.
        Every suspicion score event in chronological order.
        This is the complete, deterministic audit path from first interview to verdict.
        """
        lines = [self._header("6. REASONING TRACE")]

        if not self.verdict.suspicion_events:
            lines.append("No scoring events recorded.")
            return "\n".join(lines)

        for event in self.verdict.suspicion_events:
            delta_str = f"{event.delta:+.2f}"
            lines.append(
                f"  {event.timestamp[:19]} | {event.suspect_name:<25} | "
                f"{event.event_type:<30} | delta={delta_str} | score={event.score_after:.2f}"
            )
            lines.append(f"    Reason: {event.description}")

        return "\n".join(lines)

    def _section_contradictions(self) -> str:
        """
        Section 7 — Contradictions found.
        Every detected conflict: suspects involved, claim comparison, resolution status.
        """
        lines = [self._header("7. CONTRADICTIONS FOUND")]

        if not self.case_state.contradictions:
            lines.append("No contradictions detected during the investigation.")
            return "\n".join(lines)

        for idx, contradiction in enumerate(self.case_state.contradictions, start=1):
            resolved_str = "RESOLVED" if contradiction.resolved else "UNRESOLVED"
            lines.append(
                f"\nContradiction {idx} [{resolved_str}] | ID: {contradiction.contradiction_id}"
            )
            lines.append(f"  {contradiction.description}")
            lines.append(
                f"  Claim A ({contradiction.source_a}): "
                f"{contradiction.claim_a.subject} | "
                f"{contradiction.claim_a.predicate} | "
                f"{contradiction.claim_a.object_}"
            )
            lines.append(
                f"  Claim B ({contradiction.source_b}): "
                f"{contradiction.claim_b.subject} | "
                f"{contradiction.claim_b.predicate} | "
                f"{contradiction.claim_b.object_}"
            )

        return "\n".join(lines)

    def _section_errors(self) -> str:
        """
        Section 8 — Errors and anomalies.
        Parse failures, zero-claim sessions, and non-solved verdict statuses.
        All entries include the session ID and suspect for traceability.
        """
        lines = [self._header("8. ERRORS AND ANOMALIES")]
        errors_found = False

        # Testimony-level parse errors flagged by the Interrogator
        for testimony in self.case_state.testimonies:
            if testimony.parse_error:
                errors_found = True
                lines.append(
                    f"  [PARSE ERROR] Session {testimony.testimony_id} | "
                    f"Suspect: {testimony.suspect_name}"
                )
                lines.append(f"    Detail: {testimony.parse_error_detail}")

        # Sessions that produced zero claims across all turns
        zero_claim_sessions = [
            t for t in self.case_state.testimonies if len(t.claims) == 0
        ]
        for testimony in zero_claim_sessions:
            errors_found = True
            lines.append(
                f"  [ZERO CLAIMS] Session {testimony.testimony_id} | "
                f"Suspect: {testimony.suspect_name} | "
                f"Objective: {testimony.objective.value}"
            )
            lines.append(
                "    No structured claims were extracted from this session. "
                "The suspect may have been entirely evasive or the parser failed silently."
            )

        if self.verdict.status == "unsolvable":
            errors_found = True
            lines.append(
                f"  [UNSOLVABLE] Investigation reached its session budget "
                f"({self.verdict.total_sessions} sessions) without a confident verdict."
            )
        elif self.verdict.status == "inconclusive":
            errors_found = True
            lines.append(
                "  [INCONCLUSIVE] All suspects were interrogated but the top two "
                "suspicion scores could not be separated by the required margin."
            )

        if not errors_found:
            lines.append("No errors or anomalies recorded during this investigation.")

        return "\n".join(lines)

    def _section_computational_cost(self) -> str:
        """
        Section 9 — Computational cost.
        Sessions, turns, claims, tokens (from TokenCounter), elapsed time.
        """
        elapsed = datetime.utcnow() - self.start_time
        elapsed_str = str(elapsed).split(".")[0]

        total_turns = sum(len(t.turns) for t in self.case_state.testimonies)
        total_claims = sum(len(t.claims) for t in self.case_state.testimonies)
        token_summary = self.counter.summary()

        lines = [
            self._header("9. COMPUTATIONAL COST"),
            f"  Total sessions          : {self.verdict.total_sessions}",
            f"  Total turns             : {total_turns}",
            f"  Total claims extracted  : {total_claims}",
            f"  Total contradictions    : {len(self.case_state.contradictions)}",
            f"  Total scoring events    : {len(self.verdict.suspicion_events)}",
            f"  Elapsed time            : {elapsed_str}",
            "",
            f"  API calls made          : {token_summary['total_calls']}",
            f"  Prompt tokens           : {token_summary['prompt_tokens']}",
            f"  Completion tokens       : {token_summary['completion_tokens']}",
            f"  Total tokens            : {token_summary['total_tokens']}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM reconstruction narrative
    # ------------------------------------------------------------------

    def _reconstruction_narrative(self) -> str:
        """
        Generate a 2-3 paragraph plain-English narrative of what likely happened.

        Input to the LLM:
          - Victim and crime scene metadata
          - The identified suspect and confidence
          - Up to 10 key claims from the identified suspect
          - All detected contradictions

        Falls back to a deterministic summary if the LLM call fails.
        """
        if self.verdict.status == "unsolvable":
            return (
                "The investigation was unable to reconstruct the events of the crime. "
                "The session budget was exhausted before a confident verdict could be reached. "
                "Review the reasoning trace and errors section for details."
            )

        # Collect key claims from the identified suspect
        evidence_lines = []
        if self.verdict.suspect:
            suspect_claims = [
                c
                for t in self.case_state.testimonies
                if t.suspect_name == self.verdict.suspect
                for c in t.claims
            ]
            for claim in suspect_claims[:10]:
                time_part = f" ({claim.time_ref})" if claim.time_ref else ""
                evidence_lines.append(
                    f"- {claim.subject} | {claim.predicate} | {claim.object_}{time_part}"
                )

        contradiction_lines = [
            f"- {c.description}" for c in self.case_state.contradictions
        ]

        evidence_summary = "\n".join(evidence_lines) or "No direct claims available."
        contradiction_summary = "\n".join(contradiction_lines) or "None detected."

        prompt = (
            "You are writing the conclusion section of an official murder investigation report.\n\n"
            f"Victim: {self.case_state.victim_name}\n"
            f"Crime time: {self.case_state.crime_time}\n"
            f"Crime location: {self.case_state.crime_location}\n"
            f"Identified suspect: {self.verdict.suspect or 'Unknown'}\n"
            f"Confidence: {self.verdict.confidence:.0%}\n"
            f"Investigation status: {self.verdict.status}\n\n"
            f"Key claims from the identified suspect:\n{evidence_summary}\n\n"
            f"Contradictions found during investigation:\n{contradiction_summary}\n\n"
            "Write a 2-3 paragraph plain-English narrative reconstructing what likely "
            "happened. Be precise. Do not speculate beyond the evidence provided. "
            "Do not mention suspicion scores or internal system details."
        )

        try:
            response = self.narrative_client.chat.completions.create(
                model=self.narrative_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=400,
            )
            self.counter.add(response.usage)
            narrative = response.choices[0].message.content.strip()
            logger.debug("Reconstruction narrative generated successfully.")
            return narrative

        except Exception as exc:
            logger.error("Reconstruction narrative LLM call failed: %s", exc)
            if self.verdict.suspect:
                return (
                    f"Based on the evidence collected, {self.verdict.suspect} is the "
                    f"most likely perpetrator in the death of {self.case_state.victim_name} "
                    f"(confidence: {self.verdict.confidence:.0%}). "
                    "A full narrative could not be generated due to an API error. "
                    "Refer to the investigation timeline and reasoning trace for details."
                )
            return (
                "A chronological reconstruction could not be generated. "
                "Refer to the investigation timeline and reasoning trace for details."
            )
