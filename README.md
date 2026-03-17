# Zurich Commuting and Delay Analyzer

Analyze Swiss public transport reliability for Zurich-area commuting using the public API at transport.opendata.ch.

## 1) Quick start with uv

Initialize the project and dependencies:

```bash
uv init .
uv venv
source .venv/bin/activate
uv add marimo pandas httpx altair
```

If dependencies are already declared in `pyproject.toml`, install/sync them with:

```bash
uv sync
```

Run the reactive marimo notebook:

```bash
uv run marimo edit app.py
```

## 2) Data pipeline

The analyzer follows this flow:

1. **Fetch**
   - Uses async `httpx.AsyncClient` requests to query `/v1/connections`.
   - Pulls routes from `Zurich HB` to selected destinations (including `Winterthur` and other major hubs).

2. **Flatten & Clean**
   - Extracts nested fields from each connection:
     - Scheduled times (`from.departure`, `to.arrival`)
     - Prognosis times (`from.prognosis.departure`, `to.prognosis.arrival`)
   - Converts timestamps to timezone-aware pandas datetimes.
   - Handles missing prognosis data as on-time (0 delay).

3. **Delay Metrics**
   - Computes departure and arrival delay in minutes.
   - Creates `delay_minutes` using arrival delay when available, otherwise departure delay.

4. **Feature Engineering**
   - Adds `is_peak_hour` (true for common commute windows).
   - Adds `day_of_week` as an ordered categorical feature.

5. **Reactive EDA**
   - Filters by destination and max transfers via marimo UI controls.
   - Renders an Altair histogram of delay distribution.
   - Displays top 5 most reliable connections (lowest average delay).

## 3) Expected initial EDA findings

- Lower-transfer routes are typically more reliable than multi-transfer alternatives.
- Peak-hour trips generally show wider delay spread.
- The median delay should remain near 0 for many direct commuter segments.
- Reliability rankings can differ by destination due to route complexity and transfer pressure.

## 4) Project files

- `app.py`: marimo reactive notebook app with modular API and transformation logic.
- `pyproject.toml`: uv-managed project metadata and dependencies.
