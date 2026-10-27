# Changelog

All notable changes to the Lisa markdown pack are documented here.

## [Unreleased]

### Fixed

#### `forge_document` never reports success for an empty or unstated forge

- **What:** a first pass that returned visible-empty content at a clean
  `finish_reason="stop"` was persisted as an empty document and reported as
  success. `_forge_document_body` now gates on empty visible content and
  retries once at the operation's flat ceiling (or the model's own ceiling
  when that binds), and the operation reports a loud error when the retry is
  also empty. A call carrying no `query`, `existing_document`,
  `document_path`, or `target_length` and no ambient query is refused
  outright.
- **Why:** the turn treated the forge as satisfied while the user got an
  empty artifact instead of an error.

#### A zero-argument forge no longer overwrites an existing default document

- **What:** a call that does not name `document_path` still forges to
  `docs/forge.md` when nothing is there, but now refuses with
  `status="error"` — before recall and before any completion call — when a
  document already exists at that path. Naming `document_path` explicitly,
  or passing `existing_document`, keeps the overwrite legal; named
  non-default destinations are unaffected.
- **Why:** the 2026-10-26 incident: a schema-valid zero-argument call fell
  back to the ambient turn query and replaced an existing `docs/forge.md`
  with invented content. The ambient fallback itself is the documented
  "serve this turn" contract and is unchanged — the destructive act was
  overwriting a path the caller never named, so the guard keys on the
  destination, not the query.
- **What:** the refusal text is authored Markdown
  (`prompts/forge_document_unnamed_overwrite.md`), loaded through
  `host.read_authored_text` and failing loudly when the file is missing.
