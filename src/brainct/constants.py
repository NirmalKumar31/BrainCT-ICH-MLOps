"""Names and numbers shared by training, export and serving. Changing any of these changes the
model's input contract, so the serving image and the exported model must agree on them."""

from __future__ import annotations

# Output order of every model head. "any" first, as in the RSNA competition.
LABELS: tuple[str, ...] = (
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
)
SUBTYPES: tuple[str, ...] = LABELS[1:]

# RSNA 2019 metric: weighted multi-label log loss, "any" counts twice.
LABEL_WEIGHTS: tuple[float, ...] = (2.0, 1.0, 1.0, 1.0, 1.0, 1.0)

IMAGE_SIZE = 224
# HU range kept in the int16 cache; everything outside is clipped (air / metal).
HU_MIN, HU_MAX = -1024, 3071

# (center, width) in HU. Brain, subdural (blood next to bone), bone.
WINDOWS: tuple[tuple[float, float], ...] = ((40.0, 80.0), (80.0, 200.0), (600.0, 2800.0))

# 2.5D context: slices (i-1, i, i+1), each rendered in every window -> 9 input channels.
CONTEXT = 1
IN_CHANNELS = len(WINDOWS) * (2 * CONTEXT + 1)

DISCLAIMER = "Research use only. Not a medical device. Not for diagnosis or any clinical decision."
