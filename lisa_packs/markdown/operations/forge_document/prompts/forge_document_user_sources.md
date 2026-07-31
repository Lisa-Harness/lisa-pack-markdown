# Markdown User Prompt Template — Source-Aware Consolidation

User instruction:

{user_input}

{source_block}

Recalled memory context (grounding only; do NOT abridge sources to match):

```text
{memory_context}
```

Existing document (revise this; start fresh if the placeholder says so):

```markdown
{existing_document}
```

Return the consolidated markdown document as a single fenced ```markdown block.
Preserve source detail — do not summarize.
