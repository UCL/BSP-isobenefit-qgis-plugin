# Prose review protocol

How to check user-facing prose (website pages, docs, metadata, figure captions)
before it ships. Three passes, in order; none substitutes for the others.

## Pass 1: pattern scan

Grep the changed files for the mechanically findable tells:

- em dashes used as asides (any `—` outside © lines, ranges and table
  placeholders);
- banned vocabulary: elegant, beautiful, powerful, seamless, crucially,
  importantly, delve, the whole point, full stop, the story, the journey,
  the decisive result, quietly, silently;
- "not X but Y" and "X, not Y" constructions, including in headings;
- totality quantifiers (every, all, always, everything, exactly, never) for
  case-by-case checking in pass 2.

This pass catches perhaps a third of the problems. Do not stop here.

## Pass 2: full read-through

Read every rendered line as a hostile reviewer. The classes greps cannot find:

- **Unearned totality**: a totality word survives only when it states a literal
  rule ("if every check passes"); otherwise the plain statement suffices.
- **Mirrored restatement**: the same point in a caption and its body text, in
  two bullets, or in two sentences with varied wording. Keep one; delete the
  other rather than rephrasing it.
- **Self-justification**: "deliberately", "by design", "the X is intentional".
  State the fact; if the reason matters, give the reason once, where it
  belongs.
- **Meta-commentary and signposting**: sentences that describe the text
  instead of the subject ("this page sets out", "worth stating", "in brief",
  "the full account"). A page earns its structure by its headings.
- **Colon-elaboration density**: more than one "clause: elaboration" per
  paragraph; recast the extras as sentences.
- **Doublets**: near-synonym pairs ("tight and compact", "edited, corrected,
  or swapped"). Keep the stronger word.
- **Stock disclaimers**: exploring-not-prescribing, tool-not-oracle framings
  that restate what the surrounding text already establishes.
- **Uniform rhythm**: three or more consecutive sentences of the same shape.

## Pass 3: independent read

Someone who did not write the text reads it cold, with the two lists above.
The author cannot see their own residue: pass 2 by the author reliably misses
restatements of sentences the author still holds in mind. A fresh session or a
second reader counts as independent; a re-read minutes after writing does not.

## Rules of repair

- Rewrite the sentence; never patch the flagged word with a synonym
  (dial→control→setting residue is itself a tell).
- When two sentences say one thing, delete one.
- A justification appears once, at the point where the reader needs it.

## Acceptance

No pass-1 or unmistakable pass-2 findings remain; borderline findings are
either fixed or kept with a stated reason.
