# Project context

## Objective

Study whether preserving a multi-turn conversation as one supervised fine-tuning (SFT) sequence improves model behavior compared with splitting the conversation into independent instruction/output examples.

The eventual contribution should add a narrow, configurable assistant-only loss-masking path while preserving all existing LitGPT SFT behavior.

## Research hypothesis

Preserving earlier turns gives the model conversational context that is unavailable when each turn is trained independently. Training loss only on assistant messages may improve multi-turn coherence and response quality, while avoiding direct supervision on user and system text.

This is an empirical hypothesis, not an assumed improvement. The experiment must compare behavior under matched data, compute, and evaluation conditions.

## Comparison

Current-style baseline:

```text
conversation
  -> split into instruction/output examples
  -> train each pair independently
```

Proposed strategy:

```text
conversation
  -> preserve as one sequence
  -> serialize once with a conversation prompt style
  -> train only on assistant content and assistant termination tokens
```

The objective is to determine whether retaining multi-turn conversational context during SFT improves model behavior, not merely to add another masking option.
