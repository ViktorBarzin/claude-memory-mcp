#!/usr/bin/env python3
"""Extract knowledge from Claude Code conversation transcripts and store in memory API.

Runs as a background process triggered by a Stop hook. Reads the latest
conversation JSONL, extracts facts/decisions/preferences, and stores them
via the memory HTTP API.

Usage:
    python3 extract-knowledge.py <transcript_jsonl_path>

Environment:
    MEMORY_API_URL - Memory API base URL
    MEMORY_API_KEY - Memory API key
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_URL = os.environ.get("MEMORY_API_URL", os.environ.get("CLAUDE_MEMORY_API_URL", ""))
API_KEY = os.environ.get("MEMORY_API_KEY", os.environ.get("CLAUDE_MEMORY_API_KEY", ""))

# Patterns that indicate something worth remembering
KNOWLEDGE_PATTERNS = [
    # User explicitly says "remember"
    (r'\bremember\b.*?[:\-]\s*(.+)', "preferences", 0.9),
    # Decisions made
    (r'(?:decided|decision|we\'ll go with|let\'s use|agreed)\s+(?:to\s+)?(.+)', "decisions", 0.8),
    # Preferences expressed
    (r'(?:always|never|prefer|don\'t|do not)\s+(.+)', "preferences", 0.7),
    # Project facts
    (r'(?:deployed|running|hosted)\s+(?:at|on)\s+(.+)', "projects", 0.6),
]

SEEN_FILE = Path(os.environ.get("MEMORY_SEEN_FILE", Path.home() / ".claude" / "claude-memory" / "seen_lines.json"))


def load_seen() -> set:
    if SEEN_FILE.exists():
        return set(json.loads(SEEN_FILE.read_text()))
    return set()


def save_seen(seen: set) -> None:
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Keep only last 1000 entries to prevent unbounded growth
    recent = sorted(seen)[-1000:]
    SEEN_FILE.write_text(json.dumps(recent))


def api_store(content: str, category: str, importance: float, keywords: str) -> bool:
    if not API_URL or not API_KEY:
        return False
    data = json.dumps({
        "content": content,
        "category": category,
        "importance": importance,
        "expanded_keywords": keywords,
    }).encode()
    req = urllib.request.Request(
        f"{API_URL}/api/memories",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            print(f"  Stored memory #{result.get('id', '?')}: {content[:60]}...", file=sys.stderr)
            return True
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"  Failed to store: {e}", file=sys.stderr)
        return False


def extract_from_transcript(transcript_path: str) -> list[dict]:
    """Extract knowledge candidates from conversation transcript."""
    findings = []
    seen = load_seen()

    try:
        with open(transcript_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Create a hash for dedup
                line_hash = str(hash(line))
                if line_hash in seen:
                    continue
                seen.add(line_hash)

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Only look at user messages and assistant text
                role = msg.get("role", "")
                if role not in ("user", "assistant"):
                    continue

                # Extract text content
                content = ""
                if isinstance(msg.get("content"), str):
                    content = msg["content"]
                elif isinstance(msg.get("content"), list):
                    for block in msg["content"]:
                        if isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "") + " "

                if not content or len(content) < 20:
                    continue

                # Check for knowledge patterns
                for pattern, category, importance in KNOWLEDGE_PATTERNS:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        match_text = match.strip()
                        if len(match_text) > 15 and len(match_text) < 500:
                            findings.append({
                                "content": match_text,
                                "category": category,
                                "importance": importance,
                                "keywords": " ".join(match_text.split()[:10]),
                            })

    except FileNotFoundError:
        print(f"Transcript not found: {transcript_path}", file=sys.stderr)

    save_seen(seen)
    return findings


def main():
    if len(sys.argv) < 2:
        # Try to find the latest transcript
        projects_dir = Path.home() / ".claude" / "projects"
        if projects_dir.exists():
            transcripts = sorted(projects_dir.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if transcripts:
                transcript_path = str(transcripts[0])
            else:
                print("No transcript found", file=sys.stderr)
                return
        else:
            print("Usage: extract-knowledge.py <transcript.jsonl>", file=sys.stderr)
            return
    else:
        transcript_path = sys.argv[1]

    findings = extract_from_transcript(transcript_path)

    if not findings:
        return

    print(f"Found {len(findings)} knowledge candidates", file=sys.stderr)
    stored = 0
    for finding in findings[:5]:  # Max 5 per run to avoid spam
        if api_store(finding["content"], finding["category"], finding["importance"], finding["keywords"]):
            stored += 1

    print(f"Stored {stored}/{len(findings)} memories", file=sys.stderr)


if __name__ == "__main__":
    main()
