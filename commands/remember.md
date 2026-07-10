---
allowed-tools: mcp__claude_memory__memory_store, mcp__plugin_claude-memory_claude_memory__memory_store
argument-hint: <fact to remember>
description: Store a fact in persistent memory
---

Store the provided fact in persistent memory using the memory_store MCP tool.

Use $ARGUMENTS as the content to store.
Infer an appropriate category (facts, preferences, projects, people, decisions).
If the content is longer than 1,400 characters (the hard Memory bound, ADR-0007), do NOT
store it as one memory: split it by meaning into several self-contained memories — each
understandable on its own, never "part N of M" fragments — and store each separately.
Confirm what was stored.
