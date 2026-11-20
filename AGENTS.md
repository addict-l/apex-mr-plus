# AGENTS.md — How to work in this repo

This repo uses a translator workflow: a coding agent (Codex) does the work,
Muse reads the work on GitHub and explains it in plain Chinese to Kai (the
repo owner, a student, not a full-time engineer). Kai does NOT read raw agent
logs. The pull request is the communication channel. Write for a human who
will not read the code line by line.

## Branches and PRs

- Never push directly to `codex/initial-import` or `main`.
- One task = one branch named `codex/<short-task-name>` = one pull request.
- Keep diffs small and focused. If a PR grows past one clear task, split it.
- Fill in the pull request template completely (see
  `.github/pull_request_template.md`). The four sections are mandatory:
  what changed, how it was verified, open problems, decisions needed from Kai.
- Update `PROGRESS.md` in the same PR: task status, what works now, what is
  blocked.

## Verification

- Run the full test suite before opening a PR and report the actual numbers
  (passed/failed, runtime). Never claim tests pass without running them.
- If simulation or hardware verification is not possible in this environment,
  say so explicitly instead of inventing results.
- Report real measured numbers (runtime, node counts, test counts), never
  rounded guesses.

## Commits

- Small commits with messages that say what changed and why.
- Do not commit regenerable caches, historical diagnostics, or secrets.

## Communication

- The PR description is what Kai will read (via Muse's translation). Be
  concrete: file names, function names, test commands and their outputs.
- If you are stuck, say so in the PR's "Open problems" section and stop —
  do not guess your way forward on a blocked task.
- Do not assume Kai's background: avoid unexplained jargon in PR text.
