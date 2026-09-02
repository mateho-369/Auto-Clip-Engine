"""The three language agents of the pipeline.

`controller`  Stage 1 — mechanical scene breakdown (never a rewrite)
`auto_idea`   Stage 2 — Mode B topic + Khmer script generation
`qa`          Stage 6 — reviewer: deterministic checks + an LLM tone pass

Every one of them is total: they always return something usable, because the
scheduler treats "no answer" as a fallback path rather than an error.
"""
