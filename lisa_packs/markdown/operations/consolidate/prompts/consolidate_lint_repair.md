# Consolidate Lint Repair Prompt

You are repairing a consolidated markdown document that has structural lint violations. The document was merged from multiple sources; the user's original instruction and the source priorities are in the prior messages. The current document is provided below.

Rules:

- Fix ONLY the flagged structural lint issues listed below. Do not rewrite the document wholesale, do not rephrase content, and do not add or remove sections unless a lint rule requires it.
- PRESERVE DETAIL. Do not abridge, summarize, or drop content from any source. The repair must not reduce the document's coverage of the source material.
- Preserve the document's meaning, structure, tone, and all content exactly. The output must be the same document with only the flagged issues corrected.
- Do NOT add a preamble, an apology, or any prose outside the document.
- Return the corrected document as a single fenced ```markdown block containing the full document.

Original user instruction (for context, do not re-answer it):

{user_input}

Structural lint issues to fix:

```text
{lint_issues}
```

Current document (fix only the flagged issues, return the full corrected document):

```markdown
{document_so_far}
```

Return ONLY the corrected markdown as a single fenced ```markdown block.
