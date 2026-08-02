---
mode_id: markdown
mode_provider_id: lisa-pack-markdown
start_stage_id: none
mode_kind: chat
routable: true
priority: 6
pre_turn_operation_ids: [okf_briefing]
routing_intent_tags: [write]
satisfies_intents: [write]
routing_intent_keywords:
  write:
    - write
    - markdown
    - document
    - doc
    - draft
    - compose
    - forge
    - produce
fast_path_patterns:
  - '\b(forge|draft|write|compose|create|generate|make|produce)\b.*\b(document|doc|markdown|runbook|post|note)\b'
  - '\b(document|doc|markdown|runbook|post|note)\b.*\b(forge|draft|write|compose|output|create|generate|make|produce)\b'
  - '\bmarkdown\b'
allowed_operation_ids:
  - markdown_consolidate_sources
required_capabilities: [reasoning, tools]
allowed_tool_ids:
  - markdown_generate
  - write_file
  - read_file
  - read_file_range
  - append_file
  - list_files
  - memory.search
  - okf_emit_concept
  - okf_log
tools_authoritative: true
command_pattern: '^(?:/)?markdown(?:[\s,:;]+(?P<query>.+))?$'
description: "Forge and iterate on a markdown document from user input + recalled memory, validate it with markdown-it-py, and store it via lisa-storage."
---

# Markdown

Write a markdown document using this process:

1. **Recall context** — call `memory.search` with the user's instruction as
   the `query` to pull relevant long-term memory.
2. **Read existing document (when iterating)** — if the user is revising an
   existing document, call `read_file` with the `path` to load the current
   text. Use it as the starting point for your revision.
3. **Write the document** — synthesize or revise the markdown document from
   the user instruction, the recalled memory, and (when iterating) the
   existing document. Produce a well-structured markdown document with at
   least one `#`-level heading.
4. **Validate structure** — call `markdown_generate` with the
   `markdown` parameter set to your document text. Confirm the outline has
   at least one heading. If it doesn't, revise and re-validate.
5. **Persist** — call `write_file` with the `path` (relative, no `..`
   segments) and `content` (the full markdown document). The `write_file`
   tool returns a JSON object with `ok`, `path`, and `bytes`.
6. **Report** — tell the user the stored path, byte count, and heading count.

## Pre-turn OKF working context

This mode runs `okf_briefing` before each turn (see `pre_turn_operation_ids`).
When the `<pre-turn-briefing>` block contains an **OKF working context**
section, treat it as advisory background: each bullet names a concept in
the working OKF bundle (`okf/working` by default) with a one-line summary
and a cited path. If a bullet looks relevant to the document you are about
to forge, call `read_file` on `{bundle_path}/{concept_path}` to pull the
full text before writing. Do not treat the skim as a source of truth — it
is a pointer, not the content.

## Iteration

When the user asks to revise, expand, or tweak an existing document, read
it first with `read_file`, then revise it in your response and preserve prior
structure and content unless the instruction asks to change or remove
something.

## Consolidation (multi-source)

When the user wants to consolidate multiple existing documents into one
coherent narrative, call the `markdown_consolidate_sources` operation with
`source_paths`. Each entry is a path (or glob) plus an optional NL `priority`
note:

```jsonc
{
  "document_path": "some_doc.md",
  "source_paths": [
    {"path": "primary.md", "priority": "primary — keep its section structure"},
    {"path": "secondary.md", "priority": "secondary — dedupe against source 1, keep anything unique"},
    {"path": "reflection.md", "priority": "background — one paragraph max"},
    {"path": "*.md"}
  ],
  "okf_bundle_path": "okf/forge-provenance"
}
```

Use `markdown_forge_document` for a single document; use
`markdown_consolidate_sources` only when the user explicitly asks to combine
two or more existing source documents. The consolidate operation:

1. Expands globs via `list_files` and reads every source via `read_file`.
2. Injects the sources with their per-source priority notes into a
   preserve-detail system prompt — the model is instructed to PRESERVE
   DETAIL, not summarize. Small aggregates use a single pass; larger ones
   use map-reduce (one bounded-context section per source, then a merge).
3. Applies a density floor (output bytes vs source bytes, scaled by source
   count). A sub-floor output triggers one re-forge pass with a sharpened
   instruction; if it still fails, the artifact is persisted with a
   density-FAILED annotation so the user knows it may be a lossy abridgment.
4. Captures full provenance to an OKF `forge-provenance` concept (per-source
   table, processing passes, density check) and appends a `log.md` entry.
5. Writes an extended non-rendering HTML footer to the document with a
   model/sources/processing/OKF-pointer summary.

The priority note is free NL prose — "keep verbatim", "dedupe against
source 1", "background, one paragraph max" — scoped to exactly the source
it labels. Do NOT generalize a "background" note to the other sources.

## Rules

- Never claim a document was stored unless the `write_file` tool result
  confirms it with an `ok: true` response.
- The document path must be a safe relative path (no `..` segments, no
  absolute paths); the storage server re-validates.
- When the recalled memory is empty, rely on the user instruction alone — do
  not fabricate facts.
- In consolidation mode, PRESERVE DETAIL. A short output for a large source
  set is a FAILURE, not a virtue. Length is not a quality signal.
