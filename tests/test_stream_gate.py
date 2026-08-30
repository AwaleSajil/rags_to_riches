"""Nothing internal may reach the screen, no matter how the tokens are split.

The gate's whole job is a negative: a user must never see "===CHART===" or the
JSON behind it, not even for one frame. Streaming makes that hard because the
decision to emit has to be made when "===CH" is all that has arrived, so most
of these tests feed the same text at every possible split point.
"""

import pytest

from backend.services.stream_gate import (
    MARKERS,
    MarkerGate,
    extract_blocks,
    wrap,
)


def drive(chunks) -> str:
    """Feed chunks through a gate and return everything the user would see."""
    gate = MarkerGate()
    return "".join(gate.feed(c) for c in chunks) + gate.flush()


def every_split(text: str):
    """The same text as 1-char chunks, 2-char chunks, ... and all at once.

    A gate can look perfect on whole-string input and still leak when a marker
    straddles a chunk boundary, which is the normal case for real tokens.
    """
    for size in range(1, len(text) + 1):
        yield [text[i:i + size] for i in range(0, len(text), size)]


CHART = '===CHART===\n{"data": [1, 2, 3]}\n===ENDCHART==='


# --- the negative that matters ----------------------------------------------

def test_chart_block_never_appears_however_it_is_split():
    text = f"Here is your spending.\n\n{CHART}\n\nAnything else?"
    for chunks in every_split(text):
        seen = drive(chunks)
        assert "===" not in seen, f"marker leaked with chunk size {len(chunks[0])}"
        assert '"data"' not in seen
        assert "Here is your spending." in seen
        assert "Anything else?" in seen


@pytest.mark.parametrize("opening,closing", list(MARKERS.items()))
def test_every_marker_type_is_suppressed(opening, closing):
    text = f"before {opening} secret-payload {closing} after"
    for chunks in every_split(text):
        seen = drive(chunks)
        assert "secret-payload" not in seen
        assert "===" not in seen
        assert "before" in seen and "after" in seen


def test_a_partial_marker_is_never_emitted_early():
    """The core case: mid-stream, all we have is a prefix."""
    gate = MarkerGate()
    assert gate.feed("Total spend. ") == "Total spend. "
    # Every prefix of the opening marker must be withheld.
    for char in "===CHART==":
        assert gate.feed(char) == ""
    assert gate.feed("=") == ""  # marker now complete — still nothing
    assert gate.feed('{"data": []}') == ""
    assert gate.feed("===ENDCHART===") == ""
    assert gate.feed(" Done.") == " Done."


def test_two_blocks_in_one_answer():
    text = f"a {CHART} b ===IMAGES===[\"x.png\"]===ENDIMAGES=== c"
    for chunks in every_split(text):
        seen = drive(chunks)
        assert "===" not in seen
        assert "x.png" not in seen
        assert "a " in seen and " b " in seen and " c" in seen


def test_unterminated_block_is_dropped_not_flushed():
    """A truncated answer is exactly when flushing the remainder would dump
    raw JSON on screen."""
    seen = drive(["Here you go. ", '===CHART===', '{"data": [1,2'])
    assert seen == "Here you go. "


# --- and the positive: ordinary text still gets through ----------------------

def test_plain_text_passes_through_unchanged():
    text = "You spent $42.10 at Trader Joe's on Tuesday."
    for chunks in every_split(text):
        assert drive(chunks) == text


def test_equals_signs_that_are_not_markers_survive():
    """Markdown rules and code must not be eaten by the marker heuristic."""
    text = "Heading\n=======\nx == y, and a === b in JS."
    for chunks in every_split(text):
        assert drive(chunks) == text


def test_text_ending_mid_prefix_is_released_on_flush():
    """'===CH' that never becomes a marker is real text and must not vanish."""
    assert drive(["done ", "===CH"]) == "done ===CH"


def test_nothing_in_nothing_out():
    assert drive([]) == ""
    assert drive(["", "", ""]) == ""


def test_gate_reports_when_it_is_inside_a_block():
    gate = MarkerGate()
    gate.feed("text ===CHART===")
    assert gate.suppressing is True
    gate.feed("===ENDCHART===")
    assert gate.suppressing is False


# --- extraction: the same job on the final, complete answer -------------------
#
# The gate protects the live view; extract_blocks produces the authoritative
# content the client keeps. Both must strip exactly the same things, so a block
# hidden while streaming can never reappear in the final message.

@pytest.mark.parametrize("opening", list(MARKERS))
def test_a_block_is_removed_and_its_payload_returned(opening):
    content, payloads = extract_blocks(
        f"before {wrap(opening, ' payload ')} after", opening
    )
    assert payloads == ["payload"]  # stripped, so json.loads works on it
    assert content == "before  after"
    assert "===" not in content


def test_several_blocks_of_one_kind_come_back_in_order():
    text = f"a {wrap('===CHART===', '1')} b {wrap('===CHART===', '2')} c"
    content, payloads = extract_blocks(text, "===CHART===")
    assert payloads == ["1", "2"]
    assert content == "a  b  c"


def test_extraction_leaves_other_marker_kinds_alone():
    """Each kind is pulled out by its own pass, so one must not eat another."""
    text = f"{wrap('===CHART===', 'fig')}{wrap('===CONFIRM_TX===', 'tx')}"
    content, charts = extract_blocks(text, "===CHART===")
    assert charts == ["fig"]
    content, txs = extract_blocks(content, "===CONFIRM_TX===")
    assert txs == ["tx"]
    assert content == ""


def test_an_unterminated_block_keeps_the_text_but_drops_the_marker():
    """A truncated answer: the block never completed, so the tail is as likely
    to be prose as JSON. The marker must still never reach the screen."""
    content, payloads = extract_blocks("Here you go. ===CHART=== oh dear", "===CHART===")
    assert payloads == []
    assert "===" not in content
    assert content == "Here you go.  oh dear"


def test_content_without_markers_is_untouched():
    text = "You spent $42.10, and x === y in JS."
    assert extract_blocks(text, "===CHART===") == (text, [])


def test_what_the_gate_hides_extraction_also_strips():
    """The invariant tying the two halves together."""
    answer = f"Spending is up. {CHART} Anything else?"
    streamed = drive(list(answer))
    final, payloads = extract_blocks(answer, "===CHART===")
    assert "===" not in streamed and "===" not in final
    assert '"data"' not in streamed and '"data"' not in final
    assert payloads == ['{"data": [1, 2, 3]}']


# --- the agent's working-out must not stream either ---------------------------
#
# A ReAct agent calls the model repeatedly. The turns that pick a tool often
# think out loud first ("Let me look that up..."), which is working-out, not an
# answer. money_rag decides per turn whether text is shown; these cover the
# classification that decision rests on.

class FakeChunk:
    """Stands in for an AIMessageChunk without importing the whole stack."""

    def __init__(self, content=None, tool_call_chunks=None, additional_kwargs=None):
        self.content = content
        self.tool_call_chunks = tool_call_chunks or []
        self.additional_kwargs = additional_kwargs or {}


def test_a_tool_call_chunk_is_recognised():
    from money_rag import _chunk_calls_tool

    assert _chunk_calls_tool(FakeChunk(tool_call_chunks=[{"name": "sql", "args": "{}"}]))


def test_a_gemini_style_function_call_is_recognised():
    """Not every provider fills tool_call_chunks; missing this shape would let
    tool-selection prose stream as if it were the answer."""
    from money_rag import _chunk_calls_tool

    assert _chunk_calls_tool(
        FakeChunk(additional_kwargs={"function_call": {"name": "sql"}})
    )


def test_ordinary_text_is_not_mistaken_for_a_tool_call():
    from money_rag import _chunk_calls_tool

    assert not _chunk_calls_tool(FakeChunk(content="You spent $42."))


def test_text_is_read_from_a_plain_string_chunk():
    from money_rag import _chunk_text

    assert _chunk_text(FakeChunk(content="hello")) == "hello"


def test_text_is_read_from_typed_content_blocks():
    from money_rag import _chunk_text

    chunk = FakeChunk(content=[
        {"type": "text", "text": "You spent "},
        {"type": "tool_use", "input": {"q": "select 1"}},
        {"type": "text", "text": "$42."},
    ])
    # Only the text blocks; a tool_use block is not part of the answer.
    assert _chunk_text(chunk) == "You spent $42."


def test_a_chunk_with_no_content_yields_nothing():
    from money_rag import _chunk_text

    assert _chunk_text(FakeChunk()) == ""
