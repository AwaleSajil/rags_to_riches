"""Hold back anything the user should never see, while text streams past.

The agent's answer carries machine-readable blocks the chat UI strips before
display — a Plotly figure, image URLs, a transaction awaiting confirmation:

    Here is your spending.

    ===CHART===
    {"data": [...], "layout": {...}}
    ===ENDCHART===

Buffering the whole answer made that easy: the router split the markers out and
sent clean text. Streaming does not get that luxury. Tokens arrive a few
characters at a time, so "===CH" can be all we have when the decision to emit
has to be made, and emitting it means the user watches "===CHART===" and a wall
of JSON appear on screen and then vanish.

This gate solves it by never emitting a character until it is certain the
character is not part of a marker. Two rules:

  1. Any trailing text that could still grow into an opening marker is held.
  2. Once an opening marker completes, everything is suppressed until its
     closing marker arrives.

The cost is latency bounded by the longest marker (a couple of dozen
characters), which is invisible at streaming speed. The `final` event still
carries the authoritative, fully-stripped content, so the client replaces
whatever streamed — this gate protects the live view, it is not the last word.
"""

from __future__ import annotations

# The marker vocabulary, in one place. Everything that writes a block (money_rag,
# the MCP tools), hides one mid-stream (MarkerGate), or pulls one back out
# (routers/chat) works from these names, so adding a block type is one edit here
# plus its producer — not four files that have to agree by hand.
CHART = "===CHART==="
IMAGES = "===IMAGES==="
CONFIRM_TX = "===CONFIRM_TX==="
CONFIRM_FIX = "===CONFIRM_FIX==="

# Opening marker -> the closing marker that ends its suppressed region.
MARKERS: dict[str, str] = {
    CHART: "===ENDCHART===",
    IMAGES: "===ENDIMAGES===",
    CONFIRM_TX: "===ENDCONFIRM_TX===",
    CONFIRM_FIX: "===ENDCONFIRM_FIX===",
}

_LONGEST_OPENING = max(len(m) for m in MARKERS)


def wrap(opening: str, payload: str) -> str:
    """Build a marker block for an answer. The counterpart to extract_blocks."""
    return f"{opening}{payload}{MARKERS[opening]}"


def extract_blocks(content: str, opening: str) -> tuple[str, list[str]]:
    """Take every `opening`…closing block out of `content`.

    Returns the content with those blocks removed, and their payloads in the
    order they appeared. This runs on the FINAL answer, where the whole string
    is in hand — MarkerGate is the streaming counterpart, which has to make the
    same decision a few characters at a time.

    An opening with no closing marker is a truncated answer. Only the marker
    itself is dropped and the text after it is kept: at that point the block
    never completed, so the remainder is as likely to be the model's prose as it
    is to be half a JSON object, and silently deleting the tail of a reply is
    the worse failure.
    """
    closing = MARKERS[opening]
    payloads: list[str] = []
    while opening in content:
        before, rest = content.split(opening, 1)
        if closing not in rest:
            content = before + rest
            break
        payload, after = rest.split(closing, 1)
        payloads.append(payload.strip())
        content = before + after
    return content, payloads


class MarkerGate:
    """Feed it streamed text; it returns only what is safe to show."""

    def __init__(self) -> None:
        self._held = ""
        # The closing marker being waited for, or None when not suppressing.
        self._closing: str | None = None

    @property
    def suppressing(self) -> bool:
        return self._closing is not None

    def feed(self, text: str) -> str:
        """Return the portion of everything seen so far that is safe to emit."""
        self._held += text
        emitted: list[str] = []

        while True:
            if self._closing is not None:
                index = self._held.find(self._closing)
                if index == -1:
                    # Still inside the block. Keep only enough tail to notice a
                    # closing marker split across two chunks.
                    keep = len(self._closing) - 1
                    self._held = self._held[-keep:] if keep else ""
                    break
                self._held = self._held[index + len(self._closing):]
                self._closing = None
                continue

            opening = self._earliest_opening()
            if opening is not None:
                index, marker = opening
                emitted.append(self._held[:index])
                self._held = self._held[index + len(marker):]
                self._closing = MARKERS[marker]
                continue

            # No complete opening marker present. Emit everything except a tail
            # that could still become one.
            hold = self._partial_opening_length()
            if hold:
                emitted.append(self._held[:-hold])
                self._held = self._held[-hold:]
            else:
                emitted.append(self._held)
                self._held = ""
            break

        return "".join(emitted)

    def flush(self) -> str:
        """End of stream: release text that was held but never became a marker.

        Anything still inside an unterminated block is dropped — a truncated
        block is exactly the case where showing the remainder would leak JSON.
        """
        if self._closing is not None:
            self._held = ""
            return ""
        remaining, self._held = self._held, ""
        return remaining

    # -- internals ------------------------------------------------------------

    def _earliest_opening(self) -> tuple[int, str] | None:
        best: tuple[int, str] | None = None
        for marker in MARKERS:
            index = self._held.find(marker)
            if index != -1 and (best is None or index < best[0]):
                best = (index, marker)
        return best

    def _partial_opening_length(self) -> int:
        """Length of the longest suffix that is a proper prefix of some marker.

        This is what stops "===CH" reaching the screen while we wait to learn
        whether the next chunk makes it "===CHART===" or just prose about C#.
        """
        longest = min(len(self._held), _LONGEST_OPENING - 1)
        for size in range(longest, 0, -1):
            suffix = self._held[-size:]
            if any(marker.startswith(suffix) for marker in MARKERS):
                return size
        return 0
