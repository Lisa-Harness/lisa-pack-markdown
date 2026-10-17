# Markdown Expansion Prompt

A markdown document you are authoring is missing sections that its declared
section plan requires. The document-so-far and the user's original
instruction are below. The earlier section plan is authoritative — the
document is incomplete without these sections.

Rules:

- Write ONLY the missing sections listed below, nothing else. Do not
  rewrite, summarize, or repeat any part of the document-so-far.
- Start each missing section with its declared `##` heading exactly as
  planned.
- Match the structure and tone of the document-so-far: same heading depth
  under the section, matching list marker style, and fenced-code-block
  language tags.
- Ground the content in the recalled memory context and the user's
  instruction — do not pad with filler or restate the document's earlier
  sections.
- If a planned section genuinely does not apply to this document, write the
  heading with a one-sentence note saying why, instead of omitting it.
- Do NOT add a preamble, an apology, or prose outside the sections.

Original user instruction (for context, do not re-answer it):

{user_input}

Document-so-far:

```markdown
{document_so_far}
```

Missing sections to write (each under its declared heading):

{missing_sections}

Return ONLY the missing sections as a single fenced ```markdown block.