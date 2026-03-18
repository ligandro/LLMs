# Step 1 (2–3 Weeks): Programming Foundations

## Outcome
Build strong Python and engineering basics so every later AI/agent project is clean, maintainable, and testable.

## Topics to Cover
- Python fundamentals (data types, loops, functions, OOP basics)
- APIs and HTTP requests
- Async programming (`asyncio`, concurrency basics)
- JSON and data handling
- Modular and clean code practices

## Weekly Plan
### Week 1
- Complete Python fundamentals with short coding drills daily.
- Practice file I/O, dictionaries/lists, and function design.
- Start using `venv`, `pip`, and `requirements.txt` in every mini-project.

### Week 2
- Learn `requests` and build scripts that call public APIs.
- Parse and validate JSON responses.
- Add error handling (`try/except`, status code checks, retries).

### Week 3 (optional, if taking 3 weeks)
- Learn async with `asyncio` and `aiohttp`.
- Refactor scripts into modules and reusable helper functions.
- Add basic tests with `pytest`.

## Hands-On Projects
1. **API Data Collector**
   - Pull data from 2 public APIs.
   - Normalize into a single JSON schema.
   - Save output to disk and log failures.

2. **Async API Harvester**
   - Fetch 50+ endpoints concurrently.
   - Compare sync vs async runtime.
   - Add retries and timeout controls.

## Engineering Checklist
- Use a project structure like:
  - `src/`
  - `tests/`
  - `README.md`
  - `requirements.txt`
- Write functions that do one thing.
- Avoid hardcoded secrets (use `.env`).
- Add type hints for function signatures.

## Completion Criteria
- You can design small Python projects without tutorial copy-paste.
- You can consume APIs reliably with error handling.
- You can process nested JSON and build reusable modules.
- You can explain when async is useful and implement basic async workflows.

## Suggested Deliverable
Create a GitHub repo: **`python-api-foundations`** containing both projects, tests, and a setup guide.
