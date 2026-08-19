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


def nugget_bank_test_name(method, judge, execution):
    return (
        f"nugget-bank-test-{method}-{judge}-{execution['dataset']}-"
        f"{execution['llm-prompt']['name']}"
    )


def prompt_cache_download_test_name(judge, execution):
    return (
        f"download-prompt-cache-{judge}-{execution['dataset']}-"
        f"{execution['llm-prompt']['name']}"
    )


def prompt_cache_verify_test_name(judge, execution):
    return (
        f"verify-prompt-cache-{judge}-{execution['dataset']}-"
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


def configured_prompt_cache_rows(matrix):
    rows = []
    for judge, configuration in matrix["judges"].items():
        for execution in configuration.get("executions", []):
            rows.append(
                {
                    "judge": judge,
                    "llm": execution["llm-prompt"]["name"],
                    "dataset": execution["dataset"],
                    "download_test_name": prompt_cache_download_test_name(
                        judge, execution
                    ),
                    "verify_test_name": prompt_cache_verify_test_name(
                        judge, execution
                    ),
                }
            )
    return rows


def configured_nugget_bank_rows(matrix):
    nugget_bank_config = matrix.get("nugget-banks", {})
    methods = nugget_bank_config.get("tests-methods", [])
    judges = nugget_bank_config.get("to-verify", [])
    rows = []

    for judge in judges:
        if judge not in matrix["judges"]:
            raise ValueError(
                f"Nugget-bank configuration references unknown judge {judge!r}"
            )
        for execution in matrix["judges"][judge].get("executions", []):
            for method in methods:
                rows.append(
                    {
                        "judge": judge,
                        "llm": execution["llm-prompt"]["name"],
                        "dataset": execution["dataset"],
                        "method": method,
                        "test_name": nugget_bank_test_name(
                            method, judge, execution
                        ),
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


def format_runtime(value):
    if not isinstance(value, (int, float)):
        return "n/a"
    if value < 1:
        return f"{value * 1000:.1f} ms"
    return f"{value:.1f} s"


def format_size(value):
    if not isinstance(value, (int, float)) or value < 0:
        return "n/a"

    units = ("B", "KiB", "MiB", "GiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            precision = 0 if unit == "B" else 1
            return f"{size:.{precision}f} {unit}"
        size /= 1024


def nugget_bank_result_cell(record):
    if record is None:
        return '<td class="missing" title="No result recorded">—</td>'
    if record.get("status") == "failed":
        return '<td class="failed">Failed</td>'

    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        return '<td class="failed">No evaluation</td>'

    kendall = score_range(evaluation, "Min (Kendall)", "Max (Kendall)")
    tauap = score_range(evaluation, "Min (Tauap B)", "Max (Tauap B)")
    return (
        "<td>"
        '<span class="passed"><strong>Passed</strong></span>'
        f"<span><strong>Kendall:</strong> {html.escape(format_range(kendall))}</span>"
        f"<span><strong>TauAP-B:</strong> {html.escape(format_range(tauap))}</span>"
        f"<span><strong>Runtime:</strong> "
        f"{html.escape(format_runtime(record.get('time')))}</span>"
        "</td>"
    )


def prompt_cache_result_cell(download_record, verify_record):
    if download_record is None and verify_record is None:
        return '<td class="missing" title="No result recorded">—</td>'

    if (
        download_record is None
        or verify_record is None
        or download_record.get("status") == "failed"
        or verify_record.get("status") == "failed"
    ):
        missing_steps = []
        if download_record is None or download_record.get("status") == "failed":
            missing_steps.append("download")
        if verify_record is None or verify_record.get("status") == "failed":
            missing_steps.append("verification")
        return (
            '<td class="failed"><span><strong>Failed</strong></span>'
            f"<span>{html.escape(', '.join(missing_steps).capitalize())}</span></td>"
        )

    responses = verify_record.get("responses")
    file_name = verify_record.get("file_name")
    if (
        not isinstance(responses, int)
        or responses < 1
        or not isinstance(file_name, str)
        or not file_name
    ):
        return '<td class="failed">Invalid or empty cache</td>'

    return (
        "<td>"
        '<span class="passed"><strong>Passed</strong></span>'
        f"<span><strong>Backend:</strong> {html.escape(file_name)}</span>"
        f"<span><strong>Cached responses:</strong> {responses}</span>"
        f"<span><strong>Database size:</strong> "
        f"{html.escape(format_size(verify_record.get('size')))}</span>"
        f"<span><strong>Download size:</strong> "
        f"{html.escape(format_size(download_record.get('size')))}</span>"
        f"<span><strong>Runtime:</strong> download "
        f"{html.escape(format_runtime(download_record.get('time')))}, verify "
        f"{html.escape(format_runtime(verify_record.get('time')))}</span>"
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


def render_prompt_cache_table(
    rows, download_history, verify_history, timestamps
):
    timestamp_headers = "".join(
        f'<th scope="col">{html.escape(format_timestamp(timestamp))}</th>'
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
                    prompt_cache_result_cell(
                        download_history[row["download_test_name"]].get(timestamp),
                        verify_history[row["verify_test_name"]].get(timestamp),
                    )
                    for timestamp in timestamps
                )
            else:
                cells.append('<td class="missing">—</td>')
            cells.append("</tr>")
            body.append("".join(cells))

    return f"""
      <div class="table-wrap" tabindex="0" aria-label="Prompt-cache verification over time">
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


def render_nugget_bank_table(rows, history, timestamps):
    timestamp_headers = "".join(
        f'<th scope="col">{html.escape(format_timestamp(timestamp))}</th>'
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
            if row_index == 0:
                cells.append(
                    f'<th class="software-column" scope="rowgroup" '
                    f'rowspan="{len(judge_rows)}">'
                    f"{html.escape(judge)}</th>"
                )
            cells.append(
                f'<th class="llm-column" scope="row">{html.escape(row["llm"])}</th>'
            )
            cells.append(
                f'<th class="method-column" scope="row">'
                f"{html.escape(row['method'])}</th>"
            )
            if timestamps:
                cells.extend(
                    nugget_bank_result_cell(
                        history[row["test_name"]].get(timestamp)
                    )
                    for timestamp in timestamps
                )
            else:
                cells.append('<td class="missing">—</td>')
            cells.append("</tr>")
            body.append("".join(cells))

    return f"""
      <div class="table-wrap nugget-bank-table" tabindex="0" aria-label="Nugget-bank verification over time">
        <table>
          <thead>
            <tr>
              <th class="software-column" scope="col">Nugget-bank software</th>
              <th class="llm-column" scope="col">LLM</th>
              <th class="method-column" scope="col">Verification method</th>
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
    nugget_bank_rows = configured_nugget_bank_rows(matrix)
    nugget_bank_history, nugget_bank_timestamps = health_history(
        records, {row["test_name"] for row in nugget_bank_rows}
    )
    nugget_bank_table = render_nugget_bank_table(
        nugget_bank_rows, nugget_bank_history, nugget_bank_timestamps
    )
    prompt_cache_rows = configured_prompt_cache_rows(matrix)
    prompt_cache_download_history, prompt_cache_download_timestamps = health_history(
        records, {row["download_test_name"] for row in prompt_cache_rows}
    )
    prompt_cache_verify_history, prompt_cache_verify_timestamps = health_history(
        records, {row["verify_test_name"] for row in prompt_cache_rows}
    )
    prompt_cache_timestamps = sorted(
        set(prompt_cache_download_timestamps) | set(prompt_cache_verify_timestamps)
    )
    prompt_cache_table = render_prompt_cache_table(
        prompt_cache_rows,
        prompt_cache_download_history,
        prompt_cache_verify_history,
        prompt_cache_timestamps,
    )
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    dataset_names = ", ".join(f"<code>{html.escape(dataset)}</code>" for dataset in datasets)
    dataset_label = "dataset" if len(datasets) == 1 else "datasets"
    all_timestamps = (
        timestamps + nugget_bank_timestamps + prompt_cache_timestamps
    )
    generated_at = (
        format_timestamp(max(all_timestamps))
        if all_timestamps
        else "No health checks recorded"
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
      --llm-column-width: 15rem;
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
    .software-column, .llm-column, .method-column {{
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
    .method-column {{
      left: calc(var(--software-column-width) + var(--llm-column-width));
      width: 17rem;
      min-width: 17rem;
      max-width: 17rem;
      z-index: 2;
    }}
    thead .software-column, thead .llm-column, thead .method-column {{
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
    .passed {{
      color: #1a7f37;
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
      <p>
        Each configured nugget bank is exercised through its verification
        method. A green status means the test completed and produced an
        evaluation; effectiveness scores are reported without applying an
        arbitrary pass threshold.
      </p>
      {nugget_bank_table}
    </section>

    <section>
      <h2>Status: Prompt Caches</h2>
      <p>
        A green status means the archived run downloaded successfully and its
        prompt-cache database could be opened with at least one complete cached
        response. Response counts and file sizes are informational and do not
        use an arbitrary pass threshold.
      </p>
      {prompt_cache_table}
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
        const llmColumn = container.querySelector("thead .llm-column");
        if (llmColumn) {{
          container.style.setProperty(
            "--llm-column-width",
            `${{llmColumn.getBoundingClientRect().width}}px`,
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
