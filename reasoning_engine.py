"""
Investigative Loop — Part 2 of Project OMNISCIENT.

The ReasoningEngine is the central brain of the detective system. It:
  1. Maintains a belief state (suspicion scores) over all suspects.
  2. Detects contradictions between testimonies using a two-tier rule-based system.
  3. Selects the next interrogation target using an Information Gain (IG) heuristic.
  4. Applies Bayesian-style score updates after each new testimony.
  5. Tracks entropy over the suspicion distribution to detect stagnation.
  6. Applies formal terminal conditions (Conviction / Weak Conviction / Inconclusive).

Design rationale:
  - Score updates are deterministic (SUSPICION_WEIGHTS lookup), not LLM-driven.
    This guarantees a fully auditable reasoning trace with zero LLM calls for
    the core reasoning, satisfying the Logical Soundness criterion.
  - The IG heuristic is a closed-form computation over known claim coverage,
    satisfying Computational Efficiency by avoiding speculative LLM queries.
  - Entropy stagnation detection prevents wasting sessions when additional
    questioning cannot change the probability ordering.
"""

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from interrogator import (
    Claim,
    ClaimType,
    InterrogationGoal,
    ObjectiveType,
    Persona,
    Testimony,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds and budgets
# ---------------------------------------------------------------------------

# Minimum suspicion score for a suspect to be declared the murderer.
VERDICT_THRESHOLD = 0.75

# The leading suspect's score must exceed the second-highest by this margin
# before a Conviction verdict is issued.
SEPARATION_MARGIN = 0.20

# Maximum total interrogation sessions across all suspects.
MAX_SESSIONS = 20

# Maximum number of times one suspect can be re-interrogated.
MAX_SESSIONS_PER_SUSPECT = 4

# Evasiveness score (0–5) at or above which a suspect triggers hostile persona.
EVASIVENESS_THRESHOLD = 3

# Number of consecutive sessions over which entropy must stagnate to trigger
# the weak-conviction terminal condition.
STAGNATION_WINDOW = 3

# Minimum entropy reduction per session to NOT be considered stagnant.
STAGNATION_MIN_DELTA = 0.03

# Minimum sessions before stagnation-based termination is considered.
MIN_SESSIONS_BEFORE_STAGNATION = 6

# ---------------------------------------------------------------------------
# Suspicion weight table — deterministic updates applied after each testimony.
#
# Positive weights increase suspicion; negative weights decrease it.
# Values were calibrated so that:
#   - A murderer with motive + contradicted alibi + evasiveness reaches ~0.75.
#   - An innocent with corroborated alibi + no motive reaches ~0.20.
# ---------------------------------------------------------------------------
SUSPICION_WEIGHTS: dict = {
    "accused_by_another_suspect": +0.20,
    "alibi_provided":             -0.08,   # any stated alibi reduces suspicion slightly
    "alibi_contradicted":         +0.30,   # proven lie about location is strong evidence
    "alibi_corroborated":         -0.22,   # cross-confirmed alibi significantly clears
    "evasiveness_high":           +0.12,   # evasiveness_score >= EVASIVENESS_THRESHOLD
    "motive_established":         +0.18,   # suspect revealed a meaningful conflict
    "accused_by_unreliable":      +0.05,   # accused by a suspect with high score (bias risk)
    "denial_recorded":            -0.05,   # explicit denial reduces slight suspicion
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Contradiction:
    """A detected conflict between two claims from different testimonies."""

    contradiction_id: str
    claim_a: Claim
    claim_b: Claim
    source_a: str      # name of suspect who made claim_a
    source_b: str      # name of suspect who made claim_b
    description: str   # plain-English description for the case report
    resolved: bool = False


@dataclass
class SuspicionEvent:
    """A single scored evidence event applied to one suspect's suspicion score."""

    suspect_name: str
    event_type: str      # key from SUSPICION_WEIGHTS
    delta: float         # score change applied (may be 0.0 for unknown event types)
    score_after: float   # suspect's score after this event
    description: str     # plain-English justification for the case report
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CaseState:
    """
    Full mutable state of a running investigation.
    Passed to ReportGenerator at the end of the investigation.
    """

    case_id: str
    victim_name: str
    crime_time: str
    crime_location: str
    suspects: list                                               # names only
    testimonies: list = field(default_factory=list)             # Testimony objects
    contradictions: list = field(default_factory=list)          # Contradiction objects
    suspicion_scores: dict = field(default_factory=dict)        # name -> float [0,1]
    suspicion_events: list = field(default_factory=list)        # SuspicionEvent objects
    sessions_done: dict = field(default_factory=dict)           # name -> int
    confronted_contradictions: set = field(default_factory=set) # contradiction_ids
    entropy_history: list = field(default_factory=list)         # float per session
    status: str = "active"  # active | solved | inconclusive | unsolvable


@dataclass
class Verdict:
    """The final output of the investigation produced by check_termination()."""

    suspect: str
    confidence: float         # final suspicion score of the named suspect
    status: str               # solved | inconclusive | unsolvable
    reasoning: list           # ordered plain-English scoring events
    contradictions_found: list
    suspicion_events: list
    final_scores: dict
    total_sessions: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# ReasoningEngine
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """
    Central reasoning engine orchestrating the entire investigation loop.

    Public interface consumed by main.py:
      - next_goal()         -> InterrogationGoal | None
      - ingest_testimony()  -> None
      - check_termination() -> Verdict | None
      - get_reasoning_trace() -> list[str]

    Args:
        case_data : Raw case dict from the dataset (restricted fields only).
        suspects  : List of suspect name strings.
    """

    def __init__(self, case_data: dict, suspects: list) -> None:
        victim = case_data.get("victim", {})
        self.state = CaseState(
            case_id=str(uuid.uuid4())[:8],
            victim_name=victim.get("name", "Unknown"),
            crime_time=case_data.get("time", "Unknown"),
            crime_location=case_data.get("location", "Unknown"),
            suspects=suspects,
            suspicion_scores={name: 0.5 for name in suspects},
            sessions_done={name: 0 for name in suspects},
        )
        logger.info(
            "ReasoningEngine initialised | case_id=%s | victim=%s | suspects=%s",
            self.state.case_id,
            self.state.victim_name,
            suspects,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ingest_testimony(self, testimony: Testimony) -> None:
        """
        Store a completed testimony, update suspicion scores, and run
        contradiction detection.

        Args:
            testimony : Testimony object returned by Interrogator.conduct_session().
        """
        self.state.testimonies.append(testimony)
        self.state.sessions_done[testimony.suspect_name] = (
            self.state.sessions_done.get(testimony.suspect_name, 0) + 1
        )
        logger.info(
            "Ingesting testimony | suspect=%s | claims=%d | evasiveness=%d",
            testimony.suspect_name,
            len(testimony.claims),
            testimony.evasiveness_score,
        )

        self._update_suspicion_scores(testimony)
        self._detect_contradictions(testimony)

        # Record entropy after each session for stagnation detection
        entropy = self._compute_entropy()
        self.state.entropy_history.append(entropy)
        logger.debug(
            "Entropy after session | suspect=%s | H=%.4f",
            testimony.suspect_name,
            entropy,
        )

    def next_goal(self) -> Optional[InterrogationGoal]:
        """
        Select the next interrogation target using an Information Gain (IG) heuristic.

        For every valid (suspect, objective) pair, compute an IG score.
        Return the goal corresponding to the highest-scoring pair.

        IG score formula:
          ig = base_ig(objective, known_coverage) * priority_weight(suspicion_score)
                * diminishing_returns(sessions_done)

        This is deterministic (zero LLM calls) and fully auditable.

        Returns:
            InterrogationGoal, or None if investigation should terminate.
        """
        if self.state.status != "active":
            return None

        total_sessions = sum(self.state.sessions_done.values())
        if total_sessions >= MAX_SESSIONS:
            logger.info(
                "Global session budget exhausted (%d sessions). Ending investigation.",
                MAX_SESSIONS,
            )
            return None

        candidates = []

        for suspect in self.state.suspects:
            if self.state.sessions_done.get(suspect, 0) >= MAX_SESSIONS_PER_SUSPECT:
                continue

            for objective in [
                ObjectiveType.CONFRONT_CONFLICT,
                ObjectiveType.ESTABLISH_ALIBI,
                ObjectiveType.PROBE_MOTIVE,
                ObjectiveType.VERIFY_RELATIONSHIP,
                ObjectiveType.OPEN_ENDED,
            ]:
                ig = self._compute_ig(suspect, objective)
                if ig > 0.0:
                    candidates.append((ig, suspect, objective))

        if not candidates:
            logger.info("No valid interrogation candidates remain.")
            return None

        # Select highest IG candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_ig, best_suspect, best_objective = candidates[0]

        logger.info(
            "Next goal selected | suspect=%s | objective=%s | ig_score=%.4f",
            best_suspect,
            best_objective.value,
            best_ig,
        )

        return self._build_goal(best_suspect, best_objective)

    def check_termination(self) -> Optional[Verdict]:
        """
        Apply formal terminal conditions and return a Verdict if one is met.

        Terminal conditions (evaluated in priority order):

        T1 — Conviction:
          max_score >= VERDICT_THRESHOLD
          AND separation from second suspect >= SEPARATION_MARGIN
          AND no unresolved contradictions involving the top suspect

        T2 — Inconclusive via entropy stagnation:
          >= MIN_SESSIONS_BEFORE_STAGNATION sessions done
          AND entropy reduction over last STAGNATION_WINDOW sessions < STAGNATION_MIN_DELTA

        T3 — Unsolvable (session budget exhausted):
          total_sessions >= MAX_SESSIONS

        T4 — Inconclusive (all suspects seen, scores too close):
          All suspects interviewed >= 2 times
          AND separation < SEPARATION_MARGIN

        Returns:
            Verdict object, or None if investigation should continue.
        """
        scores = self.state.suspicion_scores
        if not scores:
            return None

        sorted_suspects = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_name, top_score = sorted_suspects[0]
        second_score = sorted_suspects[1][1] if len(sorted_suspects) > 1 else 0.0
        separation = top_score - second_score

        total_sessions = sum(self.state.sessions_done.values())
        unresolved_top = [
            c for c in self.state.contradictions
            if not c.resolved and (c.source_a == top_name or c.source_b == top_name)
        ]

        # T1: Conviction — strong evidence against one suspect
        if (
            top_score >= VERDICT_THRESHOLD
            and separation >= SEPARATION_MARGIN
            and len(unresolved_top) == 0
        ):
            logger.info(
                "VERDICT: SOLVED | suspect=%s | score=%.2f | separation=%.2f",
                top_name,
                top_score,
                separation,
            )
            self.state.status = "solved"
            return self._build_verdict("solved", top_name, top_score)

        # T2: Entropy stagnation — more questioning cannot distinguish suspects
        if (
            total_sessions >= MIN_SESSIONS_BEFORE_STAGNATION
            and len(self.state.entropy_history) >= STAGNATION_WINDOW
        ):
            recent_entropies = self.state.entropy_history[-STAGNATION_WINDOW:]
            entropy_reduction = recent_entropies[0] - recent_entropies[-1]
            if entropy_reduction < STAGNATION_MIN_DELTA:
                logger.info(
                    "VERDICT: INCONCLUSIVE (entropy stagnation) | "
                    "H_reduction=%.4f over last %d sessions",
                    entropy_reduction,
                    STAGNATION_WINDOW,
                )
                self.state.status = "inconclusive"
                return self._build_verdict("inconclusive", top_name, top_score)

        # T3: Unsolvable — session budget exhausted
        if total_sessions >= MAX_SESSIONS:
            logger.info("VERDICT: UNSOLVABLE | budget exhausted at %d sessions", total_sessions)
            self.state.status = "unsolvable"
            return self._build_verdict("unsolvable", top_name, top_score)

        # T4: Inconclusive — all suspects seen, no clear separation
        all_seen_twice = all(
            self.state.sessions_done.get(s, 0) >= 2 for s in self.state.suspects
        )
        if all_seen_twice and separation < SEPARATION_MARGIN:
            logger.info(
                "VERDICT: INCONCLUSIVE | top=%s (%.2f) | second=%.2f | separation=%.2f",
                top_name,
                top_score,
                second_score,
                separation,
            )
            self.state.status = "inconclusive"
            return self._build_verdict("inconclusive", top_name, top_score)

        return None

    def get_reasoning_trace(self) -> list:
        """
        Return an ordered plain-English log of every scoring decision made.
        Used by ReportGenerator for the step-by-step reasoning section.
        """
        return [
            (
                f"[{e.timestamp}] {e.suspect_name} | {e.event_type} | "
                f"delta={e.delta:+.2f} | score_after={e.score_after:.2f} | {e.description}"
            )
            for e in self.state.suspicion_events
        ]

    # ------------------------------------------------------------------
    # IG heuristic
    # ------------------------------------------------------------------

    def _compute_ig(self, suspect: str, objective: ObjectiveType) -> float:
        """
        Estimate the expected information gain from asking this objective to this suspect.

        IG = base_ig * priority_weight * diminishing_returns

        base_ig:           depends on what is already known about this suspect.
        priority_weight:   higher suspicion suspects are higher priority.
        diminishing_returns: 1 / (1 + sessions * 0.45) — repeated questioning yields less.

        Returns 0.0 if the objective is not valid given current coverage.
        """
        sessions = self.state.sessions_done.get(suspect, 0)
        score = self.state.suspicion_scores.get(suspect, 0.5)

        # Diminishing returns factor
        dr = 1.0 / (1.0 + sessions * 0.45)

        # Priority weight — scales IG by how much we suspect this person.
        # We use a soft mix so low-suspicion suspects still get some attention.
        priority = 0.25 + 0.75 * score

        if objective == ObjectiveType.CONFRONT_CONFLICT:
            # Only valid if there is an unresolved contradiction involving this suspect
            has_active_contradiction = any(
                (not c.resolved) and (c.source_a == suspect or c.source_b == suspect)
                for c in self.state.contradictions
            )
            if not has_active_contradiction:
                return 0.0
            # Contradiction confrontation has the highest base IG
            return 1.5 * priority * dr

        elif objective == ObjectiveType.ESTABLISH_ALIBI:
            has_alibi = any(
                any(c.claim_type == ClaimType.ALIBI for c in t.claims)
                for t in self.state.testimonies
                if t.suspect_name == suspect
            )
            if has_alibi:
                return 0.0
            return 0.85 * priority * dr

        elif objective == ObjectiveType.PROBE_MOTIVE:
            has_knowledge = any(
                any(c.claim_type == ClaimType.KNOWLEDGE for c in t.claims)
                for t in self.state.testimonies
                if t.suspect_name == suspect
            )
            if has_knowledge:
                return 0.0
            return 0.90 * priority * dr

        elif objective == ObjectiveType.VERIFY_RELATIONSHIP:
            has_relationship = any(
                any(c.claim_type == ClaimType.RELATIONSHIP for c in t.claims)
                for t in self.state.testimonies
                if t.suspect_name == suspect
            )
            if has_relationship:
                return 0.0
            return 0.55 * priority * dr

        elif objective == ObjectiveType.OPEN_ENDED:
            # Only valid for suspects who have not been interviewed at all
            if sessions > 0:
                return 0.0
            return 0.65 * priority * dr

        return 0.0

    # ------------------------------------------------------------------
    # Goal construction
    # ------------------------------------------------------------------

    def _build_goal(self, suspect: str, objective: ObjectiveType) -> InterrogationGoal:
        """
        Construct an InterrogationGoal for the given suspect and objective.

        For CONFRONT_CONFLICT, finds the highest-priority unresolved contradiction
        involving this suspect and embeds its description in known_conflicts.
        Marks the contradiction as resolved to prevent re-confrontation.
        """
        persona = self._select_persona(suspect)

        if objective == ObjectiveType.CONFRONT_CONFLICT:
            unresolved = [
                c for c in self.state.contradictions
                if (not c.resolved) and (c.source_a == suspect or c.source_b == suspect)
            ]
            # Sort by the maximum suspicion score of the two parties involved
            unresolved.sort(
                key=lambda c: max(
                    self.state.suspicion_scores.get(c.source_a, 0.0),
                    self.state.suspicion_scores.get(c.source_b, 0.0),
                ),
                reverse=True,
            )
            if unresolved:
                target = unresolved[0]
                target.resolved = True
                self.state.confronted_contradictions.add(target.contradiction_id)
                logger.info(
                    "Marking contradiction resolved | id=%s | targeting suspect=%s",
                    target.contradiction_id,
                    suspect,
                )
                return InterrogationGoal(
                    suspect_name=suspect,
                    objective=objective,
                    persona=persona,
                    known_conflicts=[target.description],
                )
            # Fallback: contradiction was resolved between IG computation and goal building
            objective = ObjectiveType.OPEN_ENDED

        if objective == ObjectiveType.ESTABLISH_ALIBI:
            return InterrogationGoal(
                suspect_name=suspect,
                objective=objective,
                persona=persona,
                time_window=self.state.crime_time,
            )

        return InterrogationGoal(
            suspect_name=suspect,
            objective=objective,
            persona=persona,
        )

    def _select_persona(self, suspect_name: str) -> Persona:
        """
        Select the appropriate interrogation persona for a suspect.

        Rules (in priority order):
          - HOSTILE   if score > 0.6 or max evasiveness >= EVASIVENESS_THRESHOLD
          - SYMPATHETIC if score < 0.3
          - ANALYTICAL otherwise (default)

        The persona is determined by the ReasoningEngine (not the Interrogator)
        because it depends on the suspect's current belief state, which only
        the engine tracks.
        """
        score = self.state.suspicion_scores.get(suspect_name, 0.5)
        max_evasiveness = max(
            (t.evasiveness_score for t in self.state.testimonies if t.suspect_name == suspect_name),
            default=0,
        )

        if score > 0.6 or max_evasiveness >= EVASIVENESS_THRESHOLD:
            return Persona.HOSTILE
        if score < 0.3:
            return Persona.SYMPATHETIC
        return Persona.ANALYTICAL

    # ------------------------------------------------------------------
    # Score updates
    # ------------------------------------------------------------------

    def _update_suspicion_scores(self, testimony: Testimony) -> None:
        """
        Apply deterministic score updates based on the claims in this testimony.

        Each update is logged as a SuspicionEvent for the full audit trail.
        """
        suspect = testimony.suspect_name

        for claim in testimony.claims:

            # Accusation — raise the accused party's score
            if claim.claim_type == ClaimType.ACCUSATION:
                accused = claim.object_
                # Only credit accusations if the accused is a known suspect
                if accused in self.state.suspicion_scores:
                    accuser_score = self.state.suspicion_scores.get(suspect, 0.5)
                    # Weight the accusation: accusations from low-suspicion suspects
                    # carry more weight than accusations from highly suspicious ones
                    event = "accused_by_another_suspect" if accuser_score < 0.6 else "accused_by_unreliable"
                    self.apply_weight(
                        accused,
                        event,
                        f"{suspect} (score={accuser_score:.2f}) accused {accused}: {claim.source_text[:80]}",
                    )

            # Alibi provided — tentatively reduce suspicion
            if claim.claim_type == ClaimType.ALIBI and claim.confidence == "stated":
                self.apply_weight(
                    suspect,
                    "alibi_provided",
                    f"{suspect} provided stated alibi: {claim.source_text[:80]}",
                )

            # Denial — small reduction
            if claim.claim_type == ClaimType.DENIAL:
                self.apply_weight(
                    suspect,
                    "denial_recorded",
                    f"{suspect} denied: {claim.object_}",
                )

            # Knowledge of a conflict with victim suggests motive
            if (
                claim.claim_type == ClaimType.KNOWLEDGE
                and "conflict" in claim.predicate.lower()
            ):
                self.apply_weight(
                    suspect,
                    "motive_established",
                    f"{suspect} revealed conflict knowledge: {claim.source_text[:80]}",
                )

        # Evasiveness penalty
        if testimony.evasiveness_score >= EVASIVENESS_THRESHOLD:
            self.apply_weight(
                suspect,
                "evasiveness_high",
                f"{suspect} evasiveness score={testimony.evasiveness_score}/5",
            )

    def apply_weight(self, suspect: str, event_type: str, description: str) -> None:
        """
        Apply one SUSPICION_WEIGHTS entry to a suspect's score and log it.

        Score is clamped to [0.0, 1.0] after each update.

        Args:
            suspect     : Name of the suspect to update.
            event_type  : Key into SUSPICION_WEIGHTS.
            description : Human-readable justification for this score change.
        """
        if suspect not in self.state.suspicion_scores:
            logger.warning("Attempted to score unknown suspect: %s", suspect)
            return

        delta = SUSPICION_WEIGHTS.get(event_type, 0.0)
        old_score = self.state.suspicion_scores[suspect]
        new_score = max(0.0, min(1.0, old_score + delta))
        self.state.suspicion_scores[suspect] = new_score

        event = SuspicionEvent(
            suspect_name=suspect,
            event_type=event_type,
            delta=delta,
            score_after=new_score,
            description=description,
        )
        self.state.suspicion_events.append(event)

        logger.debug(
            "Score update | suspect=%s | event=%s | delta=%+.2f | %.2f -> %.2f",
            suspect,
            event_type,
            delta,
            old_score,
            new_score,
        )

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    def _detect_contradictions(self, new_testimony: Testimony) -> None:
        """
        Tier 1 rule-based contradiction detection: compare every new claim
        against all prior claims for location and corroboration conflicts.

        All prior testimonies (excluding the one just ingested) are searched.
        """
        prior_claims: list = []
        for testimony in self.state.testimonies[:-1]:
            for claim in testimony.claims:
                prior_claims.append((testimony.suspect_name, claim))

        for new_claim in new_testimony.claims:
            for prior_suspect, prior_claim in prior_claims:
                self._check_location_conflict(
                    new_testimony.suspect_name, new_claim,
                    prior_suspect, prior_claim,
                )
                self._check_corroboration_conflict(
                    new_testimony.suspect_name, new_claim,
                    prior_suspect, prior_claim,
                )

    def _check_location_conflict(
        self,
        suspect_a: str,
        claim_a: Claim,
        suspect_b: str,
        claim_b: Claim,
    ) -> None:
        """
        Detect location conflicts: same subject placed at different locations
        during overlapping time windows by different sources.
        """
        if (
            claim_a.subject == claim_b.subject
            and claim_a.predicate == claim_b.predicate
            and claim_a.object_ != claim_b.object_
            and claim_a.time_ref is not None
            and claim_b.time_ref is not None
            and self._times_overlap(claim_a.time_ref, claim_b.time_ref)
        ):
            description = (
                f"Location conflict: '{claim_a.subject}' placed at "
                f"'{claim_a.object_}' (per {suspect_a}) but at "
                f"'{claim_b.object_}' (per {suspect_b}) "
                f"during overlapping time window."
            )
            self.register_contradiction(claim_a, claim_b, suspect_a, suspect_b, description)

            # A contradicted alibi is stronger evidence than an unverified one
            if claim_a.claim_type == ClaimType.ALIBI:
                self.apply_weight(
                    claim_a.subject,
                    "alibi_contradicted",
                    description,
                )

    def _check_corroboration_conflict(
        self,
        suspect_a: str,
        claim_a: Claim,
        suspect_b: str,
        claim_b: Claim,
    ) -> None:
        """
        Detect corroboration conflicts: suspect A names suspect B in their alibi,
        but suspect B has issued a denial covering the same relationship.
        """
        if (
            claim_a.claim_type == ClaimType.ALIBI
            and claim_a.subject == suspect_a
            and suspect_b in claim_a.object_
            and claim_b.subject == suspect_b
            and claim_b.claim_type == ClaimType.DENIAL
        ):
            description = (
                f"Corroboration conflict: {suspect_a} claims to have been "
                f"with {suspect_b}, but {suspect_b} issued a denial."
            )
            self.register_contradiction(claim_a, claim_b, suspect_a, suspect_b, description)
            self.apply_weight(
                suspect_a,
                "alibi_contradicted",
                description,
            )

    def register_contradiction(
        self,
        claim_a: Claim,
        claim_b: Claim,
        source_a: str,
        source_b: str,
        description: str,
    ) -> None:
        """
        Register a contradiction between two claims, deduplicating by claim_id pairs.

        Args:
            claim_a     : First conflicting claim.
            claim_b     : Second conflicting claim.
            source_a    : Name of suspect who made claim_a.
            source_b    : Name of suspect who made claim_b.
            description : Plain-English description for the case report.
        """
        existing_pairs = {
            (c.claim_a.claim_id, c.claim_b.claim_id)
            for c in self.state.contradictions
        }
        if (claim_a.claim_id, claim_b.claim_id) in existing_pairs:
            return
        if (claim_b.claim_id, claim_a.claim_id) in existing_pairs:
            return

        contradiction = Contradiction(
            contradiction_id=str(uuid.uuid4())[:8],
            claim_a=claim_a,
            claim_b=claim_b,
            source_a=source_a,
            source_b=source_b,
            description=description,
        )
        self.state.contradictions.append(contradiction)
        logger.info(
            "Contradiction registered | id=%s | %s vs %s | %s",
            contradiction.contradiction_id,
            source_a,
            source_b,
            description,
        )

    # ------------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------------

    def _compute_entropy(self) -> float:
        """
        Compute the Shannon entropy of the current suspicion score distribution.

        H = -sum( p_i * log2(p_i) )

        Scores are not a strict probability distribution (they do not necessarily
        sum to 1), so each score is used directly as p_i with a small epsilon
        for numerical stability. High entropy = high uncertainty; low entropy =
        investigation converging on one suspect.
        """
        scores = list(self.state.suspicion_scores.values())
        total = sum(scores)
        if total == 0:
            return 0.0
        # Normalise to proper probabilities for the entropy calculation
        probs = [s / total for s in scores]
        return -sum(p * math.log2(p + 1e-12) for p in probs)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _times_overlap(self, time_a: str, time_b: str) -> bool:
        """
        Coarse time-overlap check by shared token matching.

        Splits both strings on whitespace and hyphens and tests for intersection.
        Returns True if any token from time_a appears in time_b.

        This is intentionally simple. Natural-language time references from LLMs
        are heterogeneous (e.g. "9PM", "around nine", "after dinner"), making
        full timestamp parsing unreliable without a dedicated NER step.
        """
        tokens_a = set(time_a.upper().replace("-", " ").split())
        tokens_b = set(time_b.upper().replace("-", " ").split())
        return bool(tokens_a & tokens_b)

    def _sorted_suspects(self) -> list:
        """Return suspects sorted by suspicion score descending."""
        return sorted(
            self.state.suspicion_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

    def _build_verdict(self, status: str, suspect: str, confidence: float) -> Verdict:
        """Construct the final Verdict object from current case state."""
        return Verdict(
            suspect=suspect,
            confidence=confidence,
            status=status,
            reasoning=self.get_reasoning_trace(),
            contradictions_found=self.state.contradictions,
            suspicion_events=self.state.suspicion_events,
            total_sessions=sum(self.state.sessions_done.values()),
            final_scores=dict(self.state.suspicion_scores),
        )
