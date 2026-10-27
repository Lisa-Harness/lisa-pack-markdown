# forge_document Usage

Invoke the `markdown_forge_document` operation with a natural-language instruction in `query`.

The operation recalls relevant memory, synthesizes a markdown document from your instruction and the recalled context, validates its structure with the markdown-it-py outline tool, and stores it via the lisa-storage `write_file` tool.

To iterate on an existing document, pass `document_path` and `existing_document` (the current text). When `existing_document` is omitted, the operation reads the current document at `document_path` from storage automatically.

An existing document at the default path (`docs/forge.md`) is never replaced by a call that did not name `document_path`: name the path, or pass `existing_document`, to revise it.

Example: `query` = "write a runbook for restarting a stuck VM from the team's past incidents"
