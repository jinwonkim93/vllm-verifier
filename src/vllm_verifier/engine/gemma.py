"""Decode Gemma's explicit thought channel, without searching prose for JSON."""


def final_content(text: str) -> str:
    prefix = "<|channel>thought\n"
    if text.startswith(prefix):
        _, separator, text = text[len(prefix) :].partition("<channel|>")
        if not separator:
            raise ValueError("Unterminated Gemma thought channel")
    if "<|channel>" in text or "<channel|>" in text:
        raise ValueError("Unexpected Gemma channel framing")
    return text
