---
name: memory-management
description: How Claude Memory manages persistent memory across sessions and compactions
---

# Memory Management

## Available MCP Tools
- `memory_store`: Save facts, preferences, decisions
- `memory_recall`: Retrieve relevant memories by context
- `memory_list`: List recent memories
- `memory_delete`: Delete a memory by ID
- `secret_get`: Retrieve decrypted content of a sensitive memory

## When to Store Memories
- User says "remember X" -> store immediately
- User shares preferences -> store with category "preferences"
- Important project context -> store with category "projects"
- Key decisions -> store with category "decisions"
- People details -> store with category "people"

## Content Bound & Links (ADR-0007)
- Memory content is hard-bounded at **1,400 characters** — the API rejects oversize
  stores/updates with HTTP 422 and the tools pre-validate with the same message.
- Split long knowledge by meaning into several **self-contained** memories, each
  understandable on its own — never "part N of M" fragments.
- On the API side, memories can be joined with typed links (`part-of`, `see-also`,
  `supersedes`, `resolved-by`); recall follows them — a superseding memory is served in
  place of the old one, and a `resolved-by` target is attached automatically.
- Recall sorts by **relevance** by default; pass `sort_by: "importance"` only when the
  importance axis is explicitly wanted.

## When to Recall Memories
- Before answering preference questions ("how do I like X?")
- When user references past conversations
- At session start (memories are injected via compaction recovery)

## Compaction Survival
Memory survives context compactions via:
1. PreCompact hook saves key memories to a marker file
2. UserPromptSubmit hook detects the marker and injects recovery context
3. SQLite database persists across all sessions
