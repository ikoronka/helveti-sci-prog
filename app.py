import marimo

__generated_with = "0.11.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import asyncio
    from datetime import time

    import altair as alt
    import httpx
    import marimo
    import pandas as pd

    return alt, asyncio, httpx, marimo, pd, time


@app.cell
def _():
    BASE_URL = "https://transport.opendata.ch/v1/connections"
    ORIGIN_STATION = "Zurich HB"
    HUB_DESTINATIONS = [
        "Winterthur",
        "Zurich Oerlikon",
        "Zug",
        "Baden",
        "Uster",
    ]
    REQUEST_LIMIT = 20
    return BASE_URL, HUB_DESTINATIONS, ORIGIN_STATION, REQUEST_LIMIT


@app.cell
def _(BASE_URL, REQUEST_LIMIT, asyncio, httpx):
    async def fetch_connections(
        from_station: str,
        to_station: str,
        *,
        limit: int = REQUEST_LIMIT,
    ) -> dict:
        params = {
            "from": from_station,
            "to": to_station,
            "limit": limit,
        }
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(BASE_URL, params=params)
            response.raise_for_status()
            return response.json()

    async def fetch_for_destinations(
        from_station: str,
        destinations: list[str],
        *,
        limit: int = REQUEST_LIMIT,
    ) -> list[dict]:
        tasks = [
            fetch_connections(from_station, destination, limit=limit)
            for destination in destinations
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        payloads = []
        for destination, result in zip(destinations, results):
            if isinstance(result, Exception):
                payloads.append(
                    {
                        "query_from": from_station,
                        "query_to": destination,
                        "connections": [],
                        "error": str(result),
                    }
                )
            else:
                payloads.append(
                    {
                        "query_from": from_station,
                        "query_to": destination,
                        "connections": result.get("connections", []),
                        "error": None,
                    }
                )
        return payloads

    return fetch_for_destinations


@app.cell
def _(pd, time):
    WEEKDAY_ORDER = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    def _safe_to_datetime(value):
        return pd.to_datetime(value, errors="coerce", utc=True)

    def _compute_delay_minutes(scheduled, prognosis) -> float:
        if pd.isna(scheduled):
            return 0.0
        if pd.isna(prognosis):
            return 0.0
        return (prognosis - scheduled).total_seconds() / 60.0

    def flatten_connections(payloads):
        rows: list[dict] = []
        for payload in payloads:
            query = payload.get("query_from", payload.get("from", ""))
            destination = payload.get("query_to", payload.get("to", ""))

            if isinstance(query, dict):
                query = (query.get("station") or {}).get("name", "")
            if isinstance(destination, dict):
                destination = (destination.get("station") or {}).get("name", "")

            connections = payload.get("connections", [])

            for connection in connections:
                from_obj = connection.get("from", {})
                to_obj = connection.get("to", {})
                prognosis_from = (from_obj.get("prognosis") or {})
                prognosis_to = (to_obj.get("prognosis") or {})

                departure_scheduled = _safe_to_datetime(from_obj.get("departure"))
                arrival_scheduled = _safe_to_datetime(to_obj.get("arrival"))
                departure_prognosis = _safe_to_datetime(
                    prognosis_from.get("departure")
                )
                arrival_prognosis = _safe_to_datetime(prognosis_to.get("arrival"))

                dep_delay = _compute_delay_minutes(
                    departure_scheduled,
                    departure_prognosis,
                )
                arr_delay = _compute_delay_minutes(
                    arrival_scheduled,
                    arrival_prognosis,
                )
                delay_minutes = arr_delay if not pd.isna(arrival_prognosis) else dep_delay

                rows.append(
                    {
                        "route_from": query,
                        "route_to": destination,
                        "departure_scheduled": departure_scheduled,
                        "arrival_scheduled": arrival_scheduled,
                        "departure_prognosis": departure_prognosis,
                        "arrival_prognosis": arrival_prognosis,
                        "transfers": len(connection.get("sections", [])) - 1,
                        "duration": connection.get("duration"),
                        "departure_delay_minutes": dep_delay,
                        "arrival_delay_minutes": arr_delay,
                        "delay_minutes": delay_minutes,
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "route_from",
                    "route_to",
                    "departure_scheduled",
                    "arrival_scheduled",
                    "departure_prognosis",
                    "arrival_prognosis",
                    "transfers",
                    "duration",
                    "departure_delay_minutes",
                    "arrival_delay_minutes",
                    "delay_minutes",
                    "is_peak_hour",
                    "day_of_week",
                ]
            )

        frame = pd.DataFrame(rows)

        departure_local = frame["departure_scheduled"].dt.tz_convert("Europe/Zurich")
        frame["is_peak_hour"] = departure_local.dt.time.apply(
            lambda t: (
                time(7, 0) <= t <= time(9, 0)
                or time(16, 0) <= t <= time(19, 0)
            )
            if pd.notna(t)
            else False
        )

        frame["day_of_week"] = pd.Categorical(
            departure_local.dt.day_name(),
            categories=WEEKDAY_ORDER,
            ordered=True,
        )
        frame["transfers"] = frame["transfers"].clip(lower=0)

        return frame

    def most_reliable_connections(frame, top_n: int = 5):
        if frame.empty:
            return pd.DataFrame(columns=["route_to", "avg_delay_minutes", "trips"])

        summary = (
            frame.groupby(["route_from", "route_to", "transfers"], as_index=False)
            .agg(
                avg_delay_minutes=("delay_minutes", "mean"),
                median_delay_minutes=("delay_minutes", "median"),
                trips=("delay_minutes", "size"),
            )
            .sort_values(["avg_delay_minutes", "median_delay_minutes", "transfers"])
            .head(top_n)
            .reset_index(drop=True)
        )
        return summary

    return flatten_connections, most_reliable_connections


@app.cell
def _(HUB_DESTINATIONS, ORIGIN_STATION, marimo):
    marimo.md("## Zurich Commuting and Delay Analyzer")
    destination_dropdown = marimo.ui.dropdown(
        options=HUB_DESTINATIONS,
        value="Winterthur",
        label="Destination Station",
    )
    transfer_slider = marimo.ui.slider(
        start=0,
        stop=4,
        value=2,
        step=1,
        label="Maximum Transfers",
    )
    marimo.hstack([destination_dropdown, transfer_slider], justify="space-between")
    return destination_dropdown, transfer_slider


@app.cell
async def _(HUB_DESTINATIONS, ORIGIN_STATION, fetch_for_destinations, flatten_connections):
    payloads = await fetch_for_destinations(
        ORIGIN_STATION, HUB_DESTINATIONS, limit=30
    )
    fetch_errors = [
        f"{payload['query_to']}: {payload['error']}"
        for payload in payloads
        if payload.get("error")
    ]
    all_connections = flatten_connections(payloads)
    return all_connections, fetch_errors


@app.cell
def _(fetch_errors, marimo):
    if fetch_errors:
        marimo.md("### Data Fetch Warnings")
        marimo.md("\\n".join([f"- {message}" for message in fetch_errors]))


@app.cell
def _(all_connections, destination_dropdown):
    selected_destination = destination_dropdown.value
    selected_data = all_connections[all_connections["route_to"] == selected_destination].copy()
    return selected_data


@app.cell
def _(selected_data, transfer_slider):
    filtered_data = selected_data[selected_data["transfers"] <= transfer_slider.value].copy()
    return filtered_data


@app.cell
def _(alt, filtered_data, marimo):
    marimo.md("### Delay Distribution")
    if filtered_data.empty:
        marimo.md("No data available for the selected filters.")
    else:
        histogram = (
            alt.Chart(filtered_data)
            .mark_bar()
            .encode(
                x=alt.X("delay_minutes:Q", bin=alt.Bin(maxbins=30), title="Delay (minutes)"),
                y=alt.Y("count():Q", title="Connections"),
                tooltip=[
                    alt.Tooltip("count():Q", title="Connections"),
                ],
            )
            .properties(height=320)
        )
        histogram


@app.cell
def _(filtered_data, marimo, most_reliable_connections):
    marimo.md("### Top 5 Most Reliable Connections")
    reliability_table = most_reliable_connections(filtered_data, top_n=5)
    if reliability_table.empty:
        marimo.md("No reliable connections found for the selected filters.")
    else:
        marimo.ui.table(reliability_table)


@app.cell
def _(filtered_data, marimo):
    marimo.md("### Filtered Data Preview")
    marimo.ui.table(
        filtered_data[
            [
                "route_from",
                "route_to",
                "departure_scheduled",
                "arrival_scheduled",
                "transfers",
                "delay_minutes",
                "is_peak_hour",
                "day_of_week",
            ]
        ].sort_values("departure_scheduled")
    )


@app.cell
def _(marimo, pd):
    marimo.md("### Demo-Safe Mock Delay Chart")

    mock_commute_data = pd.DataFrame(
        [
            {"route": "Zurich HB → Winterthur", "delay_minutes": 2, "time": "07:12"},
            {"route": "Zurich HB → Winterthur", "delay_minutes": 5, "time": "07:42"},
            {"route": "Zurich HB → Winterthur", "delay_minutes": 8, "time": "08:12"},
            {"route": "Zurich HB → Winterthur", "delay_minutes": 11, "time": "08:42"},
            {"route": "Zurich HB → Winterthur", "delay_minutes": 14, "time": "09:12"},
        ]
    )

    max_delay = marimo.ui.slider(
        start=0,
        stop=20,
        step=1,
        value=14,
        label="Max delay (minutes)",
    )
    max_delay

    return max_delay, mock_commute_data


@app.cell
def _(alt, marimo, max_delay, mock_commute_data):
    filtered_mock_data = mock_commute_data[
        mock_commute_data["delay_minutes"] <= max_delay.value
    ]

    chart = (
        alt.Chart(filtered_mock_data)
        .mark_bar()
        .encode(
            x=alt.X("time:N", title="Time"),
            y=alt.Y("delay_minutes:Q", title="Delay (minutes)"),
            tooltip=["route:N", "time:N", "delay_minutes:Q"],
        )
        .properties(height=280)
    )

    marimo.vstack([max_delay, chart, marimo.ui.table(filtered_mock_data)])


if __name__ == "__main__":
    app.run()
