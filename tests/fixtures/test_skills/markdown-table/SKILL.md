---
name: markdown-table
description: Convert tabular data (CSV, TSV, list-of-dicts, list-of-tuples) into a clean GitHub-flavored Markdown table. Use when a user provides rows of data and asks for a Markdown table, or when an existing table needs to be reformatted.
---

To convert tabular data to a Markdown table, follow these steps. The output
target is GitHub-flavored Markdown, which most chat surfaces and code
review tools render correctly; the formatting choices below are the ones
that survive a roundtrip through those renderers without surprising
behavior.

First, infer the column header row. If the input has an explicit header (a
CSV with a header line, a list of dicts whose keys are the column names),
use those names verbatim. If the input is a list of tuples or rows without
a header, ask the user for column names rather than inventing them; an
incorrectly inferred header is much harder to spot later than a missing
one is now. Strip surrounding whitespace from each header cell and
collapse interior runs of whitespace to a single space, so a column whose
source was "  Last  Name " becomes "Last Name".

Second, escape pipe characters and newlines inside cells. A literal pipe
inside a cell terminates the cell early in Markdown's table grammar, so it
must be replaced with the HTML entity `&#124;` or with an escaped pipe
`\|`. A literal newline inside a cell breaks the table structure entirely,
so it must be replaced with `<br>`. Do not try to preserve multi-line cell
content; flatten it. If the source data legitimately needs line breaks
inside a cell, add a footnote below the table rather than embedding raw
newlines.

Third, compute column widths and pad. For each column, find the widest
cell after escaping; this width determines the column's rendered width.
Pad every cell in that column with trailing spaces to match the widest
width. Padding is purely cosmetic — Markdown renderers ignore it — but it
makes the source table readable in plain text, which matters when the
table is read in a terminal or a diff. Left-align text columns and
right-align numeric columns by setting the alignment row appropriately:
`:---` for left, `---:` for right, `:---:` for center. Choose alignment
per column, not per cell.

Fourth, emit the rendered table in this exact shape. The first line is
the header row, surrounded by pipes and one space of padding on each
side. The second line is the alignment row using the alignment markers
above. Each subsequent line is one data row, also surrounded by pipes and
one space of padding. The closing pipe at the end of each line is
required for portability across renderers; some renderers tolerate its
absence, but several major ones (including the one used by some chat
surfaces) drop the final column entirely without it.

If the table has more than twenty rows, consider whether a Markdown table
is the right output at all. Long tables are awkward to scroll through and
hard to filter visually; a CSV download link or a summary statistic plus a
small representative excerpt is often a better answer.
