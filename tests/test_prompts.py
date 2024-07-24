# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import yaml

import litgpt.config
from litgpt import Config
from litgpt.prompts import (
    Alpaca,
    Default,
    Llama3,
    PromptStyle,
    has_prompt_style,
    load_prompt_style,
    prompt_styles,
    save_prompt_style,
)


def test_default_prompt_style(mock_tokenizer):
    prompt_style = Default()
    prompt = "This is a test prompt."
    assert prompt_style.apply(prompt) == prompt
    assert prompt_style.stop_tokens(mock_tokenizer) == ([mock_tokenizer.eos_id],)


def test_prompt_style_from_name():
    for style_name in prompt_styles:
        assert isinstance(PromptStyle.from_name(style_name), prompt_styles[style_name])


def test_prompt_style_from_config():
    model_names = [
        "stablelm-tuned-alpha-3b",
        "stablelm-tuned-alpha-7b",
        "stablelm-zephyr-3b",
        "stablecode-instruct-alpha-3b",
        "falcon-7b-instruct",
        "falcon-40b-instruct",
        "vicuna-7b-v1.3",
        "vicuna-13b-v1.3",
        "vicuna-33b-v1.3",
        "vicuna-7b-v1.5",
        "vicuna-7b-v1.5-16k",
        "vicuna-13b-v1.5",
        "vicuna-13b-v1.5-16k",
        "longchat-7b-16k",
        "longchat-13b-16k",
        "Nous-Hermes-llama-2-7b",
        "Nous-Hermes-13b",
        "Nous-Hermes-Llama2-13b",
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf",
        "Llama-2-70b-chat-hf",
        "Llama-3-8B-Instruct",
        "Llama-3-70B-Instruct",
        "Llama-3.1-405B-Instruct",
        "Gemma-2b-it",
        "Gemma-7b-it",
        "FreeWilly2",
        "CodeLlama-7b-Instruct-hf",
        "CodeLlama-13b-Instruct-hf",
        "CodeLlama-34b-Instruct-hf",
        "CodeLlama-70b-Instruct-hf",
        "phi-1_5",
        "phi-2",
        "Phi-3-mini-4k-instruct",
        "Mistral-7B-Instruct-v0.1",
        "Mistral-7B-Instruct-v0.2",
        "tiny-llama-1.1b-chat",
        "Llama-2-7b-chat-hf-function-calling-v2",
    ]
    for template in ("RedPajama-INCITE-{}-3B-v1", "RedPajama-INCITE-7B-{}", "RedPajama-INCITE-{}-7B-v0.1"):
        model_names.append(template.format("Chat"))
        model_names.append(template.format("Instruct"))
    for c in litgpt.config.platypus:
        model_names.append(c["name"])

    for model_name in model_names:
        # by asserting the returned style is not the Default, we show that at least one of the regex patterns matched
        assert not isinstance(PromptStyle.from_config(Config.from_name(model_name)), Default)


def test_apply_prompts():
    prompt = "Is a coconut a nut or a fruit?"
    inp = "Optional input"

    for style in prompt_styles.values():
        output = style().apply(prompt, input=inp)
        assert prompt in output
        if isinstance(style, Alpaca):
            assert inp in output


class CustomPromptStyle(PromptStyle):
    def apply(self, prompt, **kwargs):
        return prompt


def test_save_load_prompt_style(tmp_path):
    # Save and load a built-in style
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    assert not has_prompt_style(checkpoint_dir)
    save_prompt_style("alpaca", checkpoint_dir)
    assert has_prompt_style(checkpoint_dir)
    with open(checkpoint_dir / "prompt_style.yaml", "r", encoding="utf-8") as file:
        contents = yaml.safe_load(file)
    assert contents == {"class_path": "litgpt.prompts.Alpaca"}
    loaded = load_prompt_style(checkpoint_dir)
    assert isinstance(loaded, Alpaca)

    # Save a custom style
    checkpoint_dir = tmp_path / "custom"
    checkpoint_dir.mkdir()
    save_prompt_style(CustomPromptStyle(), checkpoint_dir)
    with open(checkpoint_dir / "prompt_style.yaml", "r", encoding="utf-8") as file:
        contents = yaml.safe_load(file)
    assert contents == {"class_path": "tests.test_prompts.CustomPromptStyle"}
    loaded = load_prompt_style(checkpoint_dir)
    assert isinstance(loaded, CustomPromptStyle)


def test_multiturn_prompt():
    prompt = "What is the capital of France?"
    msgs = [{"role": "user", "content": prompt}]
    style = Llama3()
    simple_output = style.apply(prompt)
    multiturn_output = style.apply(msgs)
    assert simple_output == multiturn_output

    # override system prompt
    msgs = [
        {"role": "system", "content": "You are not a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    with_system_multiturn_output = style.apply(msgs)
    assert "You are not a helpful assistant." in with_system_multiturn_output

    # use default system prompt
    msgs = [
        {"role": "user", "content": prompt},
    ]
    wo_system_multiturn_output = style.apply(msgs)
    assert "You are a helpful assistant." in wo_system_multiturn_output

    # Longer turn
    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant for travel tips and recommendations"},
        {"role": "user", "content": "What is France's capital?"},
        {"role": "assistant", "content": "Bonjour! The capital of France is Paris!"},
        {"role": "user", "content": "What can I do there?"},
    ]
    multiturn_output = style.apply(msgs)

    assert multiturn_output == """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Bonjour! The capital of France is Paris!<|eot_id|><|start_header_id|>user<|end_header_id|>

What can I do there?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    # Longer list without "system"
    msgs = [
        {"role": "user", "content": "What is France's capital?"},
        {"role": "assistant", "content": "Bonjour! The capital of France is Paris!"},
        {"role": "user", "content": "What can I do there?"},
    ]
    multiturn_output = style.apply(msgs)

    assert multiturn_output == """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Bonjour! The capital of France is Paris!<|eot_id|><|start_header_id|>user<|end_header_id|>

What can I do there?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    # {random} string format shouldn't lead to key error
    content = "this is {random} {system} {user}"
    msgs = [
        {"role": "user", "content": content}
    ]
    output = style.apply(msgs)
    simple_output = style.apply(content)
    assert output == simple_output
