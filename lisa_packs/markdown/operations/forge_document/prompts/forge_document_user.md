# Markdown User Prompt Template

User instruction:

{user_input}

Recalled memory context:

```text
{memory_context}
```

Existing document (revise this; start fresh if the placeholder says so):

```markdown
{existing_document}
```

{target_length_guidance}
Before writing the document, declare your section plan: emit one line,
`<!-- forge-plan: ## Section One / ## Section Two / ... -->`, as the FIRST
line inside the fenced block, listing every `##` section the finished
document will contain. Then write exactly those sections.

Return the forged markdown document as a single fenced ```markdown block.
