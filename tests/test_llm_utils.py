import pytest

from speech_to_speech.LLM.utils import remove_unspeechable, split_sentences


def test_remove_unspeechable_normalizes_smart_apostrophes() -> None:
    assert remove_unspeechable("I’ll reply if here’s the plan.") == "I'll reply if here's the plan."


def test_remove_unspeechable_keeps_text_and_drops_emoji() -> None:
    assert remove_unspeechable("Hello 👋 lobster 🦞") == "Hello  lobster "


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Basic boundaries (the cases the streaming pipeline relies on).
        ("Hello. How are you?", ["Hello.", "How are you?"]),
        ("Let me check.", ["Let me check."]),
        ("Let me check. This may take a second.", ["Let me check.", "This may take a second."]),
        ("Hallo. Wie geht es dir?", ["Hallo.", "Wie geht es dir?"]),
        ("First one. Second one. Third one.", ["First one.", "Second one.", "Third one."]),
        # Streaming: a trailing fragment without a terminator stays as the last element
        # so callers can keep buffering it.
        ("Hello. How are yo", ["Hello.", "How are yo"]),
        ("No boundary yet", ["No boundary yet"]),
        ("", []),
        ("   ", []),
        # Must NOT split on decimals, initials, or common abbreviations.
        ("It costs 3.14 dollars. Really.", ["It costs 3.14 dollars.", "Really."]),
        ("Dr. Smith is here.", ["Dr. Smith is here."]),
        ("Mr. and Mrs. Jones left.", ["Mr. and Mrs. Jones left."]),
        ("Use e.g. this one. Then stop.", ["Use e.g. this one.", "Then stop."]),
        ("J. R. R. Tolkien wrote it.", ["J. R. R. Tolkien wrote it."]),
        ("The U.S. economy grew. Fast.", ["The U.S. economy grew.", "Fast."]),
        # Repeated terminators and ellipsis.
        ("Wait!! Are you sure?? Yes.", ["Wait!!", "Are you sure??", "Yes."]),
        ("Well... I guess so.", ["Well...", "I guess so."]),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected
