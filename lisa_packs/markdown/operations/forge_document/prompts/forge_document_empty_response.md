# Markdown Error: Empty Response

The model returned no visible content{diagnostics}.

Reasoning models bill their chain-of-thought against the completion budget:
when the budget is exhausted on thinking before any document text is
emitted, the visible response is empty (`finish_reason=length`, or a clean
stop with only reasoning content). This is a budget-shape failure, not an
input failure — refining the instruction will not help. Retry with a larger
`target_length` (it drives the token budget) or omit `target_length` to use
the configured flat budget.
