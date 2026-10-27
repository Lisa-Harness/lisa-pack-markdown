# Markdown Error: Refused Unnamed Overwrite

A document already exists at `{document_path}`, and this call did not name a
`document_path`. Overwriting a document nobody asked to replace would destroy
approved content, so nothing was written and the model was never called.

Pass `document_path` explicitly to overwrite it:

- To revise the existing document in place, pass `document_path` =
  `{document_path}` (the operation reads it first and revises rather than
  replacing it), or pass `existing_document` with the current text.
- To write somewhere else, pass the destination you want in `document_path`.
