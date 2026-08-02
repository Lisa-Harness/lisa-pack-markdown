# Markdown User Prompt Template — Per-Source Consolidation Section

User instruction:

{user_input}

## Source {source_index} — `{source_path}` (priority: {priority})

```markdown
{source_text}
```

Recalled memory context (grounding only; do NOT abridge the source to
match):

```text
{memory_context}
```

Return the markdown section for this source as a single fenced
```markdown block. Preserve source detail — do not summarize.