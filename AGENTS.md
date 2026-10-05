# Agent instructions

Before changing anything, read `DESIGN.md` (what UbuKit is and why) and
`CONTRIBUTING.md` (how to change it, the checks, and when a change is done).
This file adds only what is specific to agents.

## Boundaries

Unless the maintainer asks for it:

- do not push tags, publish packages, merge pull requests, or change
  repository settings, environments or secrets;
- do not add dependencies or CI actions.

Never:

- relax a test, a tolerance or the fixtures to make a change pass;
- add network, file or process access to the library;
- work around a principle in `DESIGN.md` instead of proposing to change it.

## Working

- Reviews cover every file in scope, or a scope agreed first, and say what
  was read. Confirm a suspicion by running code before reporting it.
- Keep the repository in English: code, comments, documents and commit
  messages.
