# Contributing to VULGARIS

Thank you for your interest in contributing. VULGARIS is an open research project
and welcomes contributions of all kinds.

## Quick start

```bash
git clone https://github.com/vulgaris-ai/vulgaris
cd vulgaris
pip install -e ".[dev]"
pytest tests/unit/ -v
```

## Ways to contribute

- **Bug reports** — open a GitHub Issue with a minimal reproducer
- **Bug fixes** — open a PR with a failing test that your fix resolves
- **New modules** — discuss in Issues first before large changes
- **Benchmarks** — add datasets or baselines to `benchmarks/`
- **Documentation** — improve docstrings or README examples

## Code style

```bash
black vulgaris/ --line-length 100
ruff check vulgaris/
```

All code must pass the CI matrix before merging.

## Adding a new module

1. Create `vulgaris/modules/your_module.py` inheriting from `vulgaris.engine.Module`
2. Add it to `vulgaris/modules/__init__.py`
3. Write tests in `tests/unit/test_modules.py`
4. Document the forward signature and expected input/output shapes

## Testing

```bash
# Unit tests (fast, no GPU needed)
pytest tests/unit/ -v

# Full test including network telecom test
pytest tests/ -v --timeout=300
```

## Commits

Use conventional commit format:

```
feat: add new anomaly detection head
fix: correct gradient accumulation in SSSR
docs: add streaming inference example
test: add edge case for empty batch
```

## Pull Request checklist

- [ ] Tests pass (`pytest tests/unit/`)
- [ ] New functionality has tests
- [ ] Imports follow the `vulgaris.*` namespace
- [ ] No `print()` statements in library code (use logging)
- [ ] Float32 dtype throughout (no float64)

## License

By contributing, you agree your contributions are licensed under Apache 2.0.
