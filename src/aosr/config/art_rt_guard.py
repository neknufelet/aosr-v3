"""SSOT constants for the ``art_rt`` reverberation-time term guard + render keep-mask.

SCORING-ARTRT-1 (PO-ratified 2026-07-10). The ``art_rt`` scoring term now scores the
render-tail *input* T60 ``late_t60 = f(alpha)`` (raw absorption) instead of the retired
``art_rt_t60 = f(alpha + 0.02)`` proxy. Two guard knobs and one keep-mask sample rate are
promoted here as a single SSOT so they cannot drift between the term (observables.py) and
any consumer, per the repo "one concern = one SSOT file" rule.

Provenance of each constant (do NOT re-decide inline anywhere; import from here):

* ``ART_RT_T_CAP_S`` — HI-side log-compressor knee. **PO decision (2026-07-10), NO
  literature provenance** (its own Appendix-B Hard-STOP (3), distinct from the term-switch
  Hard-STOP (2)); chosen above any physical room decay on the candidate set (max late_t60
  ~3.99 s) yet below the non-physical ART corner T60 (Finding B: the ART reverb lane has no
  air absorption, so a near-rigid corner carries 10**4..10**5 s). P0.1 measured the
  compressor bit-exact below the cap in fp32, so at 8 s it is a no-op on any physical
  candidate and only tames the non-physical corner stiffness.

* ``ART_RT_GUARD_SMOOTHNESS_S`` — log-compressor softness ``s``. **P0.1-measured winner**
  (s = 1.0): identity below the cap for any s (C1 at the knee), s shapes only the
  above-cap slope ``s / (s + t60 - t_cap)`` which never underflows to zero in fp32 for any
  representable T60 (~1.5e-5 at 68197 s) — no dead neuron.

* ``RENDER_DEFAULT_SAMPLE_RATE_HZ`` — the production render default sample rate. The
  scored-set keep-mask (which art_hf bands the render tail actually synthesizes:
  ``centre * sqrt(2) < sr / 2``) is defined against THIS default constant, **not per-render**
  (SCORING-ARTRT-1 sub-decision): the term is a fixed function of the material, independent
  of what sr a particular render happens to use. This is the SSOT ``forward_field_to_ir``'s
  keyword default is also pinned to (connected_forward.py), so the mask cutoff and the
  render keep-rule cannot silently split. At 48 kHz the cutoff is 16970.56 Hz and exactly
  two art_hf bands (18245.619140625, 20480.0 Hz) are excluded from the scored set.
"""

from __future__ import annotations

# PO-decided HI-side log-compressor knee (seconds). No literature provenance — see module
# docstring. Hard-STOP (3), ratified 2026-07-10.
ART_RT_T_CAP_S: float = 8.0

# Log-compressor softness (seconds). P0.1-measured winner; identity below the cap for any s.
ART_RT_GUARD_SMOOTHNESS_S: float = 1.0

# Production render default sample rate (Hz). SSOT for both ``forward_field_to_ir``'s
# keyword default and the art_rt scored-set keep-mask cutoff. Do NOT hardcode 48000
# elsewhere for these two purposes — import this.
RENDER_DEFAULT_SAMPLE_RATE_HZ: int = 48_000
