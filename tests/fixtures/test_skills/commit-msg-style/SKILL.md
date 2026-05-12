---
name: commit-msg-style
description: Write git commit messages in the project's house style. Use when a diff or change description is provided and the user wants a commit message, PR title, or changelog line.
---

When writing commit messages, follow this style. The intent is to produce
messages that are easy to skim in `git log --oneline`, easy to parse with
shell tools, and useful in code review where reviewers see the subject
line before the body.

The subject line is a single line, no trailing period, in the imperative
mood. It is at most seventy-two characters wide so it does not wrap in
narrow terminals or pull request lists. The subject begins with a type
prefix drawn from this fixed set: feat, fix, refactor, perf, test, docs,
chore, build, ci. The type is lowercase, followed by a colon and a single
space. After the prefix, name the area being changed in two to four words,
then a colon and a single space, then a short description of the change.
For example: "feat: query planner: skip predicate pushdown for window
aggregates" or "fix: auth: unblock rotate on missing kid header".

The subject is followed by a single blank line, then the body. The body
explains the why, not the what; the diff already shows the what. Wrap the
body at seventy-two characters per line, hard-wrapped with literal
newlines rather than soft-wrapped, so it renders the same in every git
viewer. Use Markdown sparingly: bullet lists are fine for enumerating
several distinct motivations; backticks are fine for symbol names; do not
use headers, blockquotes, or images.

If the change is a bug fix, the body should reference the failure mode
that the code prior to the fix exhibited. Phrase the failure in past
tense: "Before this change, X happened when Y." Then phrase the fix in
present tense, describing the new behavior: "Now, the code does Z, which
avoids X." If the change is performance work, include the measurement
methodology and the observed delta in the body, even if approximate, so
the next person debugging a regression has a reference point.

End the body with one trailer line per metadata field, each on its own
line, separated from the body by a single blank line. The supported
trailers are: "Closes: #N" for issues this commit fully resolves; "Refs:
#N" for issues this commit relates to but does not close; "Co-Authored-By:
Name <email>" for joint authorship. Trailer keys are colon-then-space
delimited, exactly as shown. Do not invent new trailer keys; the parser
that drives the changelog generator only knows about these three.
