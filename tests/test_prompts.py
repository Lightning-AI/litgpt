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


def test_prompt_style_from_config():
    from lit_gpt import Config
    from lit_gpt.prompts import PromptStyle, Default

    model_names = {
        "stablelm-tuned-alpha-3b": "stablelm-alpha",
        "stablelm-tuned-alpha-7b": "stablelm-alpha",
        "stablelm-zephyr-3b": "stablelm-zephyr",
        "stablecode-instruct-alpha-3b": "stablecode",
        "falcon-7b-instruct": "falcon",
        "falcon-40b-instruct": "falcon",
        "vicuna-7b-v1.3": "vicuna",
        "vicuna-13b-v1.3": "vicuna",
        "vicuna-33b-v1.3": "vicuna",
        "vicuna-7b-v1.5": "vicuna",
        "vicuna-7b-v1.5-16k": "vicuna",
        "vicuna-13b-v1.5": "vicuna",
        "vicuna-13b-v1.5-16k": "vicuna",
        "longchat-7b-16k": "vicuna",
        "longchat-13b-16k": "vicuna",
        "Nous-Hermes-llama-2-7b": "nous-research",
        "Nous-Hermes-13b": "nous-research",
        "Nous-Hermes-Llama2-13b": "nous-research",
        "Llama-2-7b-chat-hf": "llama2",
        "Llama-2-13b-chat-hf": "llama2",
        "Llama-2-70b-chat-hf": "llama2",
        "Gemma-2b-it": "gemma",
        "Gemma-7b-it": "gemma",
        "FreeWilly2": "freewilly2",
        "CodeLlama-7b-Instruct-hf": "codellama",
        "CodeLlama-13b-Instruct-hf": "codellama",
        "CodeLlama-34b-Instruct-hf": "codellama",
        "CodeLlama-70b-Instruct-hf": "codellama",
        "phi-1_5": "phi-1",
        "phi-2": "phi-2",
        "Mistral-7B-Instruct-v0.1": "codellama",
        "Mistral-7B-Instruct-v0.2": "codellama",
        "tiny-llama-1.1b-chat": "tinyllama",
        "Llama-2-7b-chat-hf-function-calling-v2": "llama2-function-calling",
    }

    for model_name in model_names:
        assert not isinstance(PromptStyle.from_config(Config.from_name(model_name)), Default)


def test_apply_promts():
    from lit_gpt.prompts import prompt_styles, Alpaca

    prompt = "Is a coconut a nut or a fruit?"
    inp = "Optional input"

    for style in prompt_styles.values():
        output = style().apply(prompt, input=inp)
        assert prompt in output
        if isinstance(style, Alpaca):
            assert inp in output
