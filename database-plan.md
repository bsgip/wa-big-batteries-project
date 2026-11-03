# WEMDE database: build plan

Noah is building this himself, to learn. Claude's job is to guide him one step at a time, not to build it for him. Read this file at the start of every session, and follow **How each step runs** exactly.

## Progress

Update this list when a step is finished (ask Noah first).

**Phase 1: thin slice** (one Dispatch Solution file → database → chart)
- [x] 0. Setup: `uv add psycopg[binary] orjson pytest`, empty folders and `__init__.py` files, create the `wemde` database (time zone `Australia/Perth`), run `src/database/sql/schema.sql`. Done when `psql -d wemde -c '\dt'` lists the tables.
- [ ] 1. `run_from_file_name(file_name)` in `extract/solution.py`: file name → the `runs` row (schedule, scenario, issued_at, file_variant). Done when its test passes.
- [ ] 2. `solution_interval_rows(doc)` in `extract/solution.py`: parsed JSON → one row per `solutionData` item for `solution_intervals`, without `run_id`. Uses `.get()`, because files before Nov 2025 lack some fields. Done when its test passes on a tiny fixture file.
- [ ] 3. `load_solution_file(conn, path)` in `load.py`: in one transaction, insert the `runs` row, get `run_id` back, COPY the interval rows, record the file in `loaded_files`. Done when one sample file loads, and loading it twice doesn't duplicate anything.
- [ ] 4. Load the sample files in `data/sample_raw_data/`, then check with SQL. Done when `SELECT * FROM dispatch_actual` shows sensible prices.
- [ ] 5. A notebook that plots energy price from `dispatch_actual`. Done when there's a chart Noah can show at a check-in.

**Phase 2: the rest of the Dispatch Solution.** One extract function per table, plus one line in the loader for each: `solution_facilities`, `solution_price_setting`, `solution_contingencies` + `solution_contingency_facilities`.

**Phase 3: the full corpus.** `build_db.py`: `--rebuild` with a confirmation step, `loaded_files` skipping, a `ProcessPoolExecutor` worker that returns `(result, error)` and never raises, and separate logs for corrupt files and code errors. Do the tablespace setup on `~/bigdata` first. Test on one day, time it, then run overnight.

**Phase 4: Case Input.** Stream one case at a time with `ijson.items(f, "data.caseData.item")`. Tables, one at a time: `cases`, `case_forecasts`, `scada_tags` + `case_scada`, `case_offers` + `case_offer_tranches`, `case_unconstrained_forecasts`, `case_parameters`, `case_violation_penalties`, `case_rcoq`, `facility_standing`, `loss_factors`.

**Phase 5: checks and CSVs.** `checks.sql` (row counts per day, missing intervals, `price = Σ contributions`, correction coverage), then `facility_scada` and the other CSV folders.

## How each step runs

Do one step at a time. Never start the next step until Noah says so.

1. **Where we are.** One or two sentences: which step this is, what it builds, and how it fits into the whole (e.g. "this gives us the `runs` row; step 3 will insert it").
2. **New ideas first.** Before the spec, explain the 1–3 concepts this step needs that Noah hasn't used before (e.g. a transaction, `COPY`, `RETURNING`, a fixture). Plain language, a tiny example each, no more than a short paragraph per concept. Skip any he already knows.
3. **The spec.** File and function name, inputs, outputs, edge cases, and what "done" means. Then write the **test file and an empty function stub**: only those two, never the function body.
4. **Noah writes the code.** If he's stuck, give hints one rung at a time, and only when he asks: (1) the idea, (2) the function or tool to use, (3) pseudo-code, (4) the actual lines.
5. **Run the test.** If it fails, ask Noah what he thinks is wrong before explaining. Point to the line; don't rewrite his code.
6. **Explain it back.** Ask Noah to explain the step in 2–3 sentences, as if to a colleague. Say what's right and what's missing.
7. **Wrap up.** Suggest a one-line commit message (Noah commits), tick the step in **Progress** above (after asking), and preview the next step in one sentence.

## Rules

- Noah writes the meaningful code. Claude may write plumbing (setup commands, the COPY helper, the parallel runner) only after Noah understands what it does, and only when he asks.
- Show any change to `schema.sql` as a diff and wait for approval before applying it.
- Keep replies short and plain. Define a term the first time it appears. One recommendation, not a menu.
- Commands that need sudo, or that create or drop databases: give them to Noah to run; don't run them.
- Facts already verified about the data, and the decisions behind the schema, are in the first context prompt. If unsure about one, ask; don't re-derive it.