#!/usr/bin/env python3

import argparse
import html
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).parent
DEFAULT_HEALTH_DATA = ROOT / "data" / "health-monitoring.jsonl"
DEFAULT_TEST_MATRIX = ROOT / "test-matrix.yml"
DEFAULT_OUTPUT = ROOT / "index.html"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the TREC AutoJudge infrastructure status page."
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="HTML output file (default: index.html)",
    )
    parser.add_argument(
        "--health-data",
        type=Path,
        default=DEFAULT_HEALTH_DATA,
        help="JSONL health-monitoring input",
    )
    parser.add_argument(
        "--test-matrix",
        type=Path,
        default=DEFAULT_TEST_MATRIX,
        help="YAML test matrix input",
    )
    return parser.parse_args()


def load_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
    return records


def load_matrix(path):
    with path.open(encoding="utf-8") as input_file:
        matrix = yaml.safe_load(input_file)

    if not isinstance(matrix, dict) or not isinstance(matrix.get("judges"), dict):
        raise ValueError(f"{path} must contain a 'judges' mapping")
    return matrix


def test_name(judge, execution):
    return (
        f"run-{judge}-{execution['dataset']}-"
        f"{execution['llm-prompt']['name']}"
    )


def configured_rows(matrix):
    rows = []
    for judge, configuration in matrix["judges"].items():
        executions = configuration.get("executions", [])
        seen_llms = set()
        for execution in executions:
            llm = execution["llm-prompt"]["name"]
            if llm in seen_llms:
                raise ValueError(
                    f"{judge} configures LLM {llm!r} more than once; "
                    "each software/LLM pair must identify one table row"
                )
            seen_llms.add(llm)
            rows.append(
                {
                    "judge": judge,
                    "llm": llm,
                    "dataset": execution["dataset"],
                    "test_name": test_name(judge, execution),
                }
            )
    return rows


def health_history(records, expected_test_names):
    history = defaultdict(dict)
    timestamps = set()

    for record in records:
        name = record.get("name")
        timestamp = record.get("timestamp")
        if name not in expected_test_names or not timestamp:
            continue
        history[name][timestamp] = record
        timestamps.add(timestamp)

    return history, sorted(timestamps)


def format_timestamp(timestamp):
    try:
        return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return timestamp


def format_score(value):
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.3f}"


def score_range(evaluation, minimum_key, maximum_key):
    minimum = evaluation.get(minimum_key)
    maximum = evaluation.get(maximum_key)
    if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
        return None
    return minimum, maximum


def format_range(score):
    if score is None:
        return "n/a"
    minimum, maximum = score
    if minimum == maximum:
        return format_score(minimum)
    return f"{format_score(minimum)}–{format_score(maximum)}"


def majority_scores(records, minimum_key, maximum_key):
    scores = Counter()
    for record in records:
        evaluation = record.get("evaluation")
        if isinstance(evaluation, dict):
            score = score_range(evaluation, minimum_key, maximum_key)
            if score is not None:
                scores[score] += 1

    if not scores:
        return set()
    highest_count = max(scores.values())
    return {score for score, count in scores.items() if count == highest_count}


def result_cell(record, majority_kendall, majority_tauap):
    if record is None:
        return '<td class="missing" title="No result recorded">—</td>'
    if record.get("status") == "failed":
        return '<td class="failed">Failed</td>'

    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        return '<td class="failed">No evaluation</td>'

    kendall = score_range(evaluation, "Min (Kendall)", "Max (Kendall)")
    tauap = score_range(evaluation, "Min (Tauap B)", "Max (Tauap B)")
    kendall_class = "majority" if kendall in majority_kendall else "deviation"
    tauap_class = "majority" if tauap in majority_tauap else "deviation"
    return (
        "<td>"
        f'<span><strong>Kendall:</strong> <span class="{kendall_class}">'
        f"{html.escape(format_range(kendall))}</span></span>"
        f'<span><strong>TauAP-B:</strong> <span class="{tauap_class}">'
        f"{html.escape(format_range(tauap))}</span></span>"
        "</td>"
    )


def render_results_table(rows, history, timestamps):
    timestamp_headers = "".join(
        f"<th scope=\"col\">{html.escape(format_timestamp(timestamp))}</th>"
        for timestamp in timestamps
    )
    if not timestamps:
        timestamp_headers = '<th scope="col">No health checks recorded</th>'

    rows_by_judge = defaultdict(list)
    for row in rows:
        rows_by_judge[row["judge"]].append(row)

    body = []
    for judge, judge_rows in rows_by_judge.items():
        for row_index, row in enumerate(judge_rows):
            cells = ["<tr>"]
            row_records = list(history[row["test_name"]].values())
            majority_kendall = majority_scores(
                row_records, "Min (Kendall)", "Max (Kendall)"
            )
            majority_tauap = majority_scores(
                row_records, "Min (Tauap B)", "Max (Tauap B)"
            )
            if row_index == 0:
                cells.append(
                    f'<th class="software-column" scope="rowgroup" '
                    f'rowspan="{len(judge_rows)}">'
                    f"{html.escape(judge)}</th>"
                )
            cells.append(
                f'<th class="llm-column" scope="row">{html.escape(row["llm"])}</th>'
            )
            if timestamps:
                cells.extend(
                    result_cell(
                        history[row["test_name"]].get(timestamp),
                        majority_kendall,
                        majority_tauap,
                    )
                    for timestamp in timestamps
                )
            else:
                cells.append('<td class="missing">—</td>')
            cells.append("</tr>")
            body.append("".join(cells))

    return f"""
      <div class="table-wrap" tabindex="0" aria-label="Reproducibility effectiveness over time">
        <table>
          <thead>
            <tr>
              <th class="software-column" scope="col">Software</th>
              <th class="llm-column" scope="col">LLM</th>
              {timestamp_headers}
            </tr>
          </thead>
          <tbody>
            {''.join(body)}
          </tbody>
        </table>
      </div>
    """


def render_page(matrix, records):
    rows = configured_rows(matrix)
    history, timestamps = health_history(
        records, {row["test_name"] for row in rows}
    )
    results_table = render_results_table(rows, history, timestamps)
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    dataset_names = ", ".join(f"<code>{html.escape(dataset)}</code>" for dataset in datasets)
    dataset_label = "dataset" if len(datasets) == 1 else "datasets"
    generated_at = (
        format_timestamp(max(timestamps)) if timestamps else "No health checks recorded"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TREC AutoJudge Infrastructure Status</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: system-ui, sans-serif;
      line-height: 1.5;
    }}
    body {{
      max-width: 90rem;
      margin: 0 auto;
      padding: 2rem;
    }}
    header, section {{
      margin-bottom: 2.5rem;
    }}
    .in-progress {{
      border-left: 0.35rem solid #bf8700;
      padding: 0.75rem 1rem;
      background: color-mix(in srgb, #bf8700 12%, transparent);
    }}
    .table-wrap {{
      --software-column-width: 16rem;
      position: relative;
      overflow-x: auto;
      max-width: 100%;
      isolation: isolate;
      scrollbar-gutter: stable;
      overscroll-behavior-x: contain;
    }}
    table {{
      border-collapse: separate;
      border-spacing: 0;
      min-width: 100%;
      font-size: 0.9rem;
      border-top: 1px solid #8c959f;
    }}
    th, td {{
      box-sizing: border-box;
      border-right: 1px solid #8c959f;
      border-bottom: 1px solid #8c959f;
      padding: 0.55rem 0.7rem;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    thead th {{
      background: color-mix(in srgb, Canvas 92%, CanvasText 8%);
    }}
    .software-column, .llm-column {{
      position: -webkit-sticky;
      position: sticky;
      background: Canvas;
      white-space: normal;
      overflow-wrap: anywhere;
    }}
    .software-column {{
      left: 0;
      width: 16rem;
      min-width: 16rem;
      max-width: 16rem;
      border-left: 1px solid #8c959f;
      z-index: 2;
    }}
    .llm-column {{
      left: var(--software-column-width);
      width: 15rem;
      min-width: 15rem;
      max-width: 15rem;
      z-index: 2;
    }}
    thead .software-column, thead .llm-column {{
      background: color-mix(in srgb, Canvas 92%, CanvasText 8%);
      z-index: 3;
    }}
    tbody th[scope="rowgroup"] {{
      background: color-mix(in srgb, Canvas 95%, CanvasText 5%);
    }}
    td > span {{
      display: block;
    }}
    .failed {{
      color: #cf222e;
      font-weight: 700;
    }}
    .majority {{
      color: #1a7f37;
    }}
    .deviation {{
      color: #cf222e;
    }}
    .missing {{
      color: #6e7781;
      text-align: center;
    }}
    .updated {{
      color: #6e7781;
    }}
  </style>
</head>
<body>
  <header>
    <h1>TREC AutoJudge Infrastructure Status</h1>
    <p class="updated">Latest recorded health check: {html.escape(generated_at)}</p>
  </header>

  <main>
    <section>
      <h2>Status: Re-Run AutoJudges from Promp Cache</h2>
      <p>
        Reproducibility effectiveness by software and LLM. Score ranges show
        the minimum and maximum effectiveness observed within each evaluation.
        Scores matching the most frequent score in their row are green;
        deviations are red.
      </p>
      <p>
        We intend to add more software systems to the status check, including
        a few selected systems from each team. The status checks are defined in
        the
        <a href="https://github.com/trec-auto-judge/infrastructure-status">TREC AutoJudge infrastructure-status repository</a>.
      </p>
      <p>
        The health checks currently run on the {dataset_names} {dataset_label}.
        We intend to add more datasets to the health checks later.
      </p>
      {results_table}
    </section>

    <section>
      <h2>Status: Nugget Banks</h2>
      <p class="in-progress">Automated health checks for nugget banks are in progress.</p>
    </section>

    <section>
      <h2>Status: Prompt Caches</h2>
      <p class="in-progress">Automated health checks for prompt caches are in progress.</p>
    </section>
  </main>
  <script>
    const positionTablesAtLatestResult = () => {{
      document.querySelectorAll(".table-wrap").forEach((container) => {{
        const softwareColumn = container.querySelector("thead .software-column");
        if (softwareColumn) {{
          container.style.setProperty(
            "--software-column-width",
            `${{softwareColumn.getBoundingClientRect().width}}px`,
          );
        }}
        container.scrollLeft = container.scrollWidth - container.clientWidth;
      }});
    }};

    window.addEventListener("load", () => {{
      requestAnimationFrame(() => {{
        requestAnimationFrame(positionTablesAtLatestResult);
      }});
      document.fonts?.ready.then(positionTablesAtLatestResult);
    }});
    window.addEventListener("resize", positionTablesAtLatestResult);
  </script>
</body>
</html>
"""


def main():
    args = parse_args()
    matrix = load_matrix(args.test_matrix)
    records = load_jsonl(args.health_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_page(matrix, records), encoding="utf-8")


if __name__ == "__main__":
    main()
