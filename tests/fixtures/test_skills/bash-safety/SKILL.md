---
name: bash-safety
description: Run a safety checklist before executing any shell command that could destroy data, modify shared state, or be hard to reverse. Use when asked whether a command is safe to run, or before recommending a command that mutates the filesystem.
---

Before running shell commands, follow this safety checklist. The checklist
is conservative on purpose: a few seconds spent confirming a command is far
cheaper than a few hours spent reconstructing deleted state, and there is
no general way to undo a mistake in a shared environment.

First, identify whether the command mutates state at all. A read-only
command (ls, cat, grep, find without -delete, git status, git log) is
always safe to run and you can skip the rest of the checklist. A command
that mutates state needs every later step.

Second, identify the blast radius. The blast radius is the set of files,
processes, or external resources the command can affect. For a local-only
mutation (touching a file in the current working tree, modifying a virtual
environment, writing to a temporary directory) the blast radius is small
and the rest of the checklist is light. For a mutation that touches a home
directory, a system path, a remote repository, a shared database, or a
production service, the blast radius is large and the rest of the
checklist is essential.

Third, confirm that the target of the command is what you intend. If the
command uses a variable expansion, expand the variable manually first and
read the literal expanded form. If the command uses a glob, list the
matched files first with `ls` or `printf %s\\n` to see exactly what would
be acted on. If the command uses a relative path, confirm the current
working directory with pwd before running it. The single most common
cause of catastrophic shell mistakes is a variable that expanded to an
empty string or an unexpected directory.

Fourth, prefer the reversible variant. Use `git rm` over `rm` for files
under version control; the change can be undone with `git restore`. Use
`mv` to a backup name (e.g., `mv path path.bak`) over `rm`; the file is
trivially recoverable. Use `--dry-run` first if the tool supports it
(rsync, git clean, find -delete, terraform). For destructive database
operations, take a backup first or run inside a transaction that you can
roll back. Reversibility buys you the freedom to learn from a mistake
without paying the full cost of it.

Fifth, when no reversible variant exists and the blast radius is large,
ask before running. The cost of asking is one round trip; the cost of
recovering from a mistaken irreversible command is unbounded. Phrase the
question with the literal command you propose to run and the directory
you propose to run it in, so the person approving has the same context
you do.
