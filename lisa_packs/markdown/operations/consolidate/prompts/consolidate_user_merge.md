# Markdown User Prompt Template — Section Merge

User instruction:

{user_input}

{sections_block}

Priority notes per source (govern how each section is handled in the
merge — see the system prompt):

{priorities_block}

Recalled memory context (grounding only):

```text
{memory_context}
```

Return the consolidated markdown document as a single fenced
```markdown block. Preserve every section's detail — do not summarize
or drop sections.