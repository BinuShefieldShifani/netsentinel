# Contributing

## Checks

Everything CI runs is in one script, so you can run the same thing locally:

```bash
ci/run_checks.sh setup   # once
ci/run_checks.sh all     # lint, C++ build + GoogleTest, pytest, training
```

GitHub Actions (`.github/workflows/ci.yml`) and Jenkins (`Jenkinsfile`) both
call this script. If a check is added, it goes in the script, not in one of
the CI files, so the two can't drift apart.

## Review workflow

The repository is set up for small, reviewable changes:

- One logical change per commit, with a message that says *why*, not only
  what. Bug fixes describe how the bug was found and add a regression test.
- With Gerrit, each commit is uploaded as a change (`git push origin
  HEAD:refs/for/main`). Jenkins, through the Gerrit Trigger plugin, builds
  every patch set and votes Verified +1 or -1. A change needs Verified +1 and
  Code-Review +2 before it is submitted. Rework goes into a new patch set of
  the same change (`git commit --amend`, keeping the Change-Id line).
- On GitHub the same rules apply to pull requests: CI must be green before
  merging.

## Changes that need extra care

- **Anything that decides whether a fix is applied automatically**
  (`would_auto_apply`, `build_action_plan`, `auto_applicable` in the
  knowledge base). Add a test for the new case, including one where the LLM
  is confidently wrong.
- **The baseline** (`features.py` and `cpp/`). The C++ and Python versions
  must stay identical; the parity tests check this. Change both together.
- **Anything that changes the reported numbers.** Rerun `python
  model_trainer.py` and update the README results in the same commit. Choose
  settings on the validation days (`experiments/select_features.py`), not on
  the held-out days.

## AI-assisted development

AI coding tools were used in this project. Their output is treated like any
other contribution: it is reviewed line by line, it needs tests, and claims
it makes (about results, standards or behaviour) are checked before they go
into the code or the README. Several bugs fixed in the history were found by
running the code rather than by reading it.
