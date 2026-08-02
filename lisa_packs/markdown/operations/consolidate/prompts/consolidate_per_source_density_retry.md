# Markdown Re-Forge — Per-Source Density Floor Violation

Your previous section for source `{source_path}` (priority: {priority}) was
{section_bytes} bytes against {source_bytes} bytes of source material
({ratio:.0%}). That is below the {min_ratio:.0%} per-source preservation
floor — your previous section summarized instead of preserving this
source's substance.

Re-forge the section for this ONE source, this time PRESERVING SOURCE
DETAIL. Your re-forged section MUST be at least **{min_section_bytes}
bytes** (roughly {min_section_words} words). If you produce less, you
have summarized this source, not preserved it — keep going until the
section carries this source's substance.

- Do NOT abridge. Do NOT compress sections to one line.
- Preserve tables, code blocks, lists, and qualifications from the source.
- If the source has headings, preserve its section structure as the
  backbone of your section. Walk through every one of the source's
  headings and emit a corresponding subsection.
- A short section for a large source is a FAILURE. Your job is
  completeness for this source, not brevity. Length is not a quality signal.
- Respect the source's priority note (in the prior messages) — it may
  legitimately grant condensation latitude ("background — one paragraph
  max"); if so, the floor does not apply and you may keep the brief
  version. Otherwise, expand.

The original instruction, the source text, and the priority note are in
the prior messages. Return ONLY the re-forged section as a single fenced
markdown block.
