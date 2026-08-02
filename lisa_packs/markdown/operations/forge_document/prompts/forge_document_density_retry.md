# Markdown Re-Forge — Density Floor Violation

Your previous output for this consolidation was {output_bytes} bytes against
{source_total_bytes} bytes of source material ({ratio:.0%}, {source_count}
sources). That is below the {min_ratio:.0%} preservation floor — your previous
output summarized instead of consolidating.

Re-forge the document, this time PRESERVING SOURCE DETAIL. Your re-forged
output MUST be at least **{min_output_bytes} bytes** (roughly
{min_output_words} words). If you produce less, you have summarized, not
consolidated — keep going until the document carries the substance of every
source.

- Do NOT abridge. Do NOT compress sections to one line.
- When sources overlap, MERGE their points — do not keep only one's and drop
  the others' unique contributions.
- Preserve tables, code blocks, and lists from the sources.
- Walk through every source's sections in order and emit a corresponding
  section in the document. Do NOT skip a source because "it repeats".
- A short output for a large source set is a FAILURE. Your job is
  completeness, not brevity. Length is not a quality signal.

The original instruction, source documents, and priority notes are in the
prior messages. Return ONLY the re-forged document as a single fenced
markdown block.
