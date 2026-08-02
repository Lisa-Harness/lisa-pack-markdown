# Markdown User Prompt Template — Rolling Accumulation

User instruction:

{user_input}

## Running document tail (last {max_headings} headings already written)

```markdown
{running_tail}
```

## Source chunk {chunk_index} of {chunk_total} — `{source_path}` (priority: {priority})

```markdown
{chunk_text}
```

Recalled memory context (grounding only; do NOT abridge the chunk to
match):

```text
{memory_context}
```

Return ONLY the new markdown content this chunk adds to the running
document as a single fenced markdown block. Preserve source detail — do
not summarize. Do not re-emit content already in the running tail.
