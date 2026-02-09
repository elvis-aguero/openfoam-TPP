# 20260207-1613-resource-clarity Clarify Build Suggestions + Memory Estimate
## Owner + Lease
- owner_session: openai/codex/2026-02-07T16:13:00-0500
- lease_expires: 2026-02-07T17:13:00-0500
## Goal / Acceptance Criteria
- Build menu clearly states whether configured parameters or suggested values will be used.
- Memory estimate avoids misleading low requests and reflects actual CPU parallelization.
## Constraints / Non-goals
- No changes to physical model or case setup logic beyond resource UI/estimation.
## Repo Touchpoints (files/dirs likely involved)
- main.py
## Plan
- Inspect build/run menu resource messaging and estimation logic.
- Update messaging to explicitly show configured vs suggested CPU settings.
- Adjust memory estimate to include per-core floor and/or actual decomposePar n_cpus.
- Summarize changes and rationale for user.
## Work Log (append-only, timestamped)
- [2026-02-07 16:13 EST] Created task file and indexed task.
## Messages (for other agents)
- None.
## Handoff (authoritative; 10–20 lines max)
- Task: Clarify whether builds use configured params vs suggestions; fix low memory estimates.
- Files: main.py for menu messaging and estimate_resources.
- Status: task created, no code changes yet.
- [2026-02-07 16:16 EST] Inspected build/run menu and estimate_resources logic in main.py.
- [2026-02-07 16:18 EST] Added _read_case_n_cpus helper, used it for run_case_oscar and memory estimation; added per-core memory floor.
- [2026-02-07 16:20 EST] Updated build menu final review to show configured parallelization, clarify suggestions are advisory, and confirm prompt includes n_cpus.
