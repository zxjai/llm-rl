---
name: setup
description: onboarding
---

# How to Use or Controbute to LLM-RL Library


## When to Use

First time user or developer

## Instructions

```bash 
uv sync

# for development 
uv sync --group dev

uv sync --all-groups
```

## Run 

```sh
source .venv/bin/activate && python llm_rl/transfer.py
```

## Other ways to run 

```sh
uv run --active 
uv run --no-sync --active 
uv run --no-project 
```