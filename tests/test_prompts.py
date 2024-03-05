# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.


def test_default_prompt_style(mock_tokenizer):
    from lit_gpt.prompts import Default

    prompt_style = Default()
    prompt = "This is a test prompt."
    assert prompt_style.apply(prompt) == prompt
    assert prompt_style.stop_tokens(mock_tokenizer) == ([mock_tokenizer.eos_id], )


def test_prompt_style_from_name():
    from lit_gpt.prompts import PromptStyle, prompt_styles

    for style_name in prompt_styles:
        assert isinstance(PromptStyle.from_name(style_name), prompt_styles[style_name])
