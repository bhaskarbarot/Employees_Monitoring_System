#!/usr/bin/env python3
"""
fusion.py — Probabilistic event fusion engine for Employee Monitor v4.

Replaces the simple frame-ratio EvidenceAcc with a time-decayed, confidence-weighted
fusion system that dramatically reduces false positives from brief gestures.

FusionEngine:
  - Maintains a rolling history of (timestamp, raw_confidence, signal_weight) tuples
  - Computes exponentially time-decayed weighted average of incoming signals
  - Confirms event when fused confidence >= conf_threshold for >= min_duration seconds
  - Hysteresis: once confirmed, stays confirmed until fused_conf drops below
    conf_threshold * HYSTERESIS_RATIO (prevents rapid on/off flicker)

Signal weights (passed by callers):
  1.0 — YOLO direct phone object detection (strongest evidence)
  0.7 — Pose wrist-near-ear (strong geometric evidence, no object required)
  0.5 — Motion stillness alone (weakest; supporting signal only)

Backward compatibility:
  FusionEngine exposes `.confirmed` property — same interface as EvidenceAcc.confirmed.
  Can coexist with EvidenceAcc; BehaviorEngine can query either.
"""

import logging
import time
from collections import deque
from typing import Deque, Optional, Tuple

_log = logging.getLogger("fusion")

HYSTERESIS_RATIO = 0.60   # confirmed → stays True until fused_conf < threshold * 0.6


class FusionEngine:
    """
    Time-decayed confidence fusion for one behavior event per track.

    Parameters
    ----------
    tau            : float  — time constant for exponential decay (seconds).
                              tau=2.0 means a signal 2s old counts for e^-1 ≈ 37%.
    conf_threshold : float  — fused confidence level to trigger "confirmed".
    min_duration   : float  — seconds the event must stay confirmed before
                              BehaviorEngine acts on it.  Prevents brief spikes.
    history_sec    : float  — how far back to keep signal history.  Signals
                              older than history_sec have negligible weight.
    """

    def __init__(self,
                 tau           : float = 2.0,
                 conf_threshold: float = 0.65,
                 min_duration  : float = 3.0,
                 history_sec   : float = 10.0) -> None:
        self.tau            = tau
        self.threshold      = conf_threshold
        self.min_duration   = min_duration
        self.history_sec    = history_sec

        # History: deque of (timestamp, raw_confidence, signal_weight)
        self._history: Deque[Tuple[float, float, float]] = deque()

        self._confirmed_since      : Optional[float] = None
        self._is_confirmed         : bool            = False
        # Tracks the time of the last MEANINGFUL signal so that recency-decay
        # works correctly even when tiny "probe" signals are added to force updates.
        self._last_meaningful_t    : float           = 0.0
        _MEANINGFUL_THRESHOLD      = 0.15   # conf * weight > this = meaningful

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, confidence: float, weight: float = 1.0) -> None:
        """
        Add a new signal sample.

        Parameters
        ----------
        confidence : float  — raw signal confidence in [0.0, 1.0].
                              0.0 = signal absent, 1.0 = maximally present.
        weight     : float  — signal weight.
                              1.0 = direct YOLO detection
                              0.7 = pose-based signal (wrist near ear)
                              0.5 = motion/stillness signal
        """
        now  = time.monotonic()
        conf = float(max(0., min(1., confidence)))
        w    = float(weight)
        self._history.append((now, conf, w))
        if conf * w > 0.15:   # meaningful threshold
            self._last_meaningful_t = now
        self._prune(now)
        self._update_confirmed(now)

    def add_bool(self, present: bool, weight: float = 1.0) -> None:
        """Convenience: add binary signal (1.0 if present, 0.0 if not)."""
        self.add(1.0 if present else 0.0, weight)

    def reset(self) -> None:
        """Clear all history and confirmed state (e.g., when track is lost)."""
        self._history.clear()
        self._confirmed_since   = None
        self._is_confirmed      = False
        self._last_meaningful_t = 0.0

    @property
    def confirmed(self) -> bool:
        """
        True if fused confidence is above threshold for >= min_duration seconds.
        Always recomputes from current time so decay works even without calling add().
        """
        self._update_confirmed(time.monotonic())
        return self._is_confirmed

    @property
    def fused_confidence(self) -> float:
        """Current time-decayed fused confidence (0.0–1.0)."""
        return self._compute_fused(time.monotonic())

    @property
    def confirmed_duration(self) -> float:
        """Seconds the event has been continuously confirmed (0.0 if not confirmed)."""
        if self._is_confirmed and self._confirmed_since is not None:
            return time.monotonic() - self._confirmed_since
        return 0.0

    # ── internal ──────────────────────────────────────────────────────────────

    def _prune(self, now: float) -> None:
        cutoff = now - self.history_sec
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _compute_fused(self, now: float) -> float:
        """
        Time-decayed confidence that correctly decays to 0 when signals stop.

        Formula: fused = weighted_avg_conf × recency_factor
          where recency_factor = exp(-(now - t_last_meaningful) / tau)
          and  t_last_meaningful = timestamp of last signal with conf*weight > 0.15

        Without the recency factor, Σ(w·c·decay)/Σ(w·decay) returns the historical
        average confidence regardless of age — old signals don't decay away.
        The recency factor forces the score to 0 when no meaningful signals arrive.
        Using _last_meaningful_t (not the last ANY signal) prevents tiny "probe"
        signals from resetting the decay clock.
        """
        if not self._history or self._last_meaningful_t == 0.0:
            return 0.0
        num = 0.0; den = 0.0
        for (t, conf, w) in self._history:
            decay = w * (2.718281828 ** (-(now - t) / self.tau))
            num  += decay * conf
            den  += decay
        avg_conf = num / max(den, 1e-9)
        recency  = 2.718281828 ** (-(now - self._last_meaningful_t) / self.tau)
        return avg_conf * recency

    def _update_confirmed(self, now: float) -> None:
        fused = self._compute_fused(now)

        if self._is_confirmed:
            # Hysteresis: stay confirmed unless drops well below threshold
            if fused < self.threshold * HYSTERESIS_RATIO:
                self._is_confirmed    = False
                self._confirmed_since = None
        else:
            if fused >= self.threshold:
                if self._confirmed_since is None:
                    self._confirmed_since = now
                # Must stay above threshold for min_duration
                if now - self._confirmed_since >= self.min_duration:
                    self._is_confirmed = True
            else:
                self._confirmed_since = None


# ══════════════════════════════════════════════════════════════════════════════
#  SLEEP FUSION  — multi-sensor weighted combination for sleeping detection
# ══════════════════════════════════════════════════════════════════════════════

class SleepFusion(FusionEngine):
    """
    Specialised fusion for sleeping detection using three independent sensors.

    pose_sleeping MUST be True for any signal to contribute (hard gate).
    If pose is absent (keypoint occlusion / model failure), face + motion
    alone contribute at reduced weight to avoid false positives.

    Signal composition per update call:
        conf = 0.6 * pose + 0.2 * face_hidden + 0.2 * body_still
    """

    def __init__(self) -> None:
        super().__init__(tau=5.0, conf_threshold=0.55, min_duration=8.0,
                         history_sec=20.0)

    def update(self, pose_sleeping: bool, face_hidden: bool,
               body_still: bool) -> None:
        if not pose_sleeping:
            # Pose required as primary gate; add 0.0 to decay existing history
            self.add(0.0, weight=1.0)
            return
        conf = 0.6 + 0.2 * face_hidden + 0.2 * body_still
        self.add(conf, weight=1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  PHONE FUSION  — multi-sensor fusion for phone usage detection
# ══════════════════════════════════════════════════════════════════════════════

class PhoneFusion(FusionEngine):
    """
    Fusion for phone usage detection combining YOLO object detection and
    pose-based wrist-to-ear proximity.

    Phone type tracking: accumulates separate evidence per type to determine
    whether the alert should be PHONE_HAND or PHONE_EAR.
    """

    def __init__(self) -> None:
        # Lower threshold than sleep — phone signals are sparser from overhead
        super().__init__(tau=1.5, conf_threshold=0.55, min_duration=2.0,
                         history_sec=8.0)
        # Separate sub-accumulators to determine dominant phone type
        self._ear_score : float = 0.0
        self._hand_score: float = 0.0

    def update_yolo(self, phone_type: Optional[str], yolo_conf: float) -> None:
        """Call when YOLO detects (or doesn't detect) a phone object."""
        if phone_type in ('hand', 'ear'):
            # Direct YOLO detection — strongest signal
            self.add(yolo_conf, weight=1.0)
            if phone_type == 'ear':
                self._ear_score  = min(1.0, self._ear_score  + 0.3)
                self._hand_score = max(0.0, self._hand_score - 0.1)
            else:
                self._hand_score = min(1.0, self._hand_score + 0.3)
                self._ear_score  = max(0.0, self._ear_score  - 0.1)
        elif phone_type == 'desk':
            # Phone seen on desk — penalise (person not holding it)
            self.add(max(0.0, yolo_conf - 0.6), weight=0.8)
        else:
            # No detection — decay
            self.add(0.0, weight=0.6)

    def update_pose(self, phone_on_ear: bool) -> None:
        """Call when PoseWorker updates wrist-near-ear status."""
        if phone_on_ear:
            # Pose ear signal — strong but not as direct as YOLO
            self.add(0.85, weight=0.7)
            self._ear_score = min(1.0, self._ear_score + 0.25)
        else:
            self.add(0.0, weight=0.4)

    @property
    def dominant_type(self) -> str:
        """Returns 'ear' or 'hand' based on accumulated evidence."""
        return 'ear' if self._ear_score >= self._hand_score else 'hand'

    def reset(self) -> None:
        super().reset()
        self._ear_score  = 0.0
        self._hand_score = 0.0
