<!-- BEGIN SUSHRUT MEMORY DIRECTIVES -->
## Sushrut Project Memory

This project participates in Sushrut's knowledge-base memory system. These directives govern memory tracking only. Preserve and follow all other project-specific instructions in this file.

### Project Memory Files

- Keep `memory/index.md` as the fast project overview.
- Keep `memory/notes/YYYY-MM-DD.md` updated during active work.
- Use `memory/notes/YYYY-MM-DD-<device>.md` if the same project is worked on from multiple devices on the same date.
- Track devices, servers, and paths in `memory/devices.md`.
- Track experiments and long-running jobs in `memory/runs.md`.
- Track durable findings in `memory/learnings.md`.
- Track decisions and their rationale in `memory/decisions.md`.
- Use `memory/commands/` for portable project-memory slash command specs when present.

### Project Memory Commands

If the user starts a prompt with a project memory command, follow the matching spec in `memory/commands/`:

- `/remember` - add a compact note to today's project memory.
- `/log` - record session work or a meaningful project change.
- `/run` - add or update an experiment, evaluation, or long-running job.
- `/decision` - record a project decision and rationale.
- `/learned` - record a durable finding or reusable lesson.
- `/status` - update the project context card.
- `/check-initialisation` - verify and align the project memory structure.

These command specs are shortcuts. They do not override project-specific instructions or this `AGENTS.md`.

### When To Update Memory

- After every meaningful experiment, update today's note.
- After every meaningful analysis result, update today's note and `memory/learnings.md`.
- After every meaningful code change or commit, record what changed and why.
- When using a new device, server, environment, or repo path, update `memory/devices.md`.
- When starting or finishing a long-running job, update `memory/runs.md`.
- When project status, blocker, latest useful result, or next action changes, update `memory/index.md`.

### What To Record

- Device/server
- Repo path
- Branch and commit when relevant
- Command/config
- Dataset or input source
- Output path
- Result summary
- Blocker
- Next action

### Style Rules

- Keep memory compact and chronological.
- Do not paste giant logs, full outputs, large tables, or raw dumps.
- Summarize the durable lesson and link to paths where evidence lives.
- Prefer clear next actions over vague observations.
- If unsure whether something matters, add it briefly to today's note and mark it as uncertain.
<!-- END SUSHRUT MEMORY DIRECTIVES -->
