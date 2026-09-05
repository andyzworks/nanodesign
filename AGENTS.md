# Repository checks

Before every commit that changes Python code, run:

```bash
python -m ruff format src tests
./scripts/check.sh
```

`scripts/check.sh` mirrors every validation command in GitHub Actions: Ruff lint,
Ruff format checking, pytest, and the shared-model CLI smoke test. Do not commit or
push if any check fails. Do not remove, skip, or weaken a check to make CI pass.
