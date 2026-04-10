# ai-csv-profiler

A Python utility that builds concise, AI-friendly JSON profiles of CSV files. Includes both a Tkinter GUI and a command-line interface, so you can explore data interactively or automate profiling in scripts.

------

## What the tool does

| Feature                          | Why it matters                                               |
| -------------------------------- | ------------------------------------------------------------ |
| **AI-Optimized JSON output**     | Gives downstream assistants (ChatGPT, Claude, etc.) exactly the fields they need. |
| **Dual UI: GUI + CLI**           | Pick the workflow that fits: point-and-click or batch processing. |
| **Smart type detection**         | Auto-recognizes numeric, text, date/time, boolean, categorical, and currency columns. |
| **Currency detection**           | Recognizes formatted currency values in any major currency (USD, EUR, GBP, JPY, and all Unicode currency symbols) and analyzes them as numeric data. |
| **Rich numeric analysis**        | Means, medians, quartiles, clipped statistics, and raw outlier values reported separately. |
| **Robust file handling**         | Tries multiple encodings and engines (BOM-aware), multiple separator strategies, warns on huge files. |
| **Thread-safe GUI**              | The UI stays responsive while the profiler works in the background. |
| **Comprehensive error handling** | Friendly messages for missing files, permission problems, malformed CSVs, etc. |

------

## Requirements

**Python 3.6+**

```bash
pip install pandas
```

### Linux: Installing Tkinter

```bash
# Debian/Ubuntu
sudo apt-get install python3-tk

# Fedora/CentOS/RHEL
sudo dnf install python3-tkinter   # or: sudo yum install tkinter
```

> **Note:** `tkinter`, `threading`, `queue`, `argparse`, `json`, `os`, `re`, `pathlib` are part of the Python standard library and need no extra installation.

------

## Installation

```bash
# Grab the latest script
curl -O https://raw.githubusercontent.com/hihipy/ai-csv-profiler/main/ai_csv_profiler.py

# (Optional) Make it directly executable
chmod +x ai_csv_profiler.py
```

------

## Usage

### GUI mode (default)

```bash
python ai_csv_profiler.py
```

1. **Select file** - Click "Browse..." and pick a CSV.
2. **Choose output** - Click "Save As..." and set the destination JSON file.
3. **Configure** - Check or uncheck *Include sample values* and *Detailed statistical analysis*.
4. **Analyze** - Press **Analyze CSV**.
5. **Results** - A JSON file is written and a short summary appears in the log pane.

### CLI mode

```bash
# Minimal - prints JSON to stdout
python ai_csv_profiler.py data.csv

# Write to a specific file
python ai_csv_profiler.py data.csv -o profile.json

# Long-form flags (identical effect)
python ai_csv_profiler.py data.csv --output analysis_results.json
```

Additional flags (shown in `--help`):

| Flag               | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| `--no-samples`     | Omit the three example values per column (smaller output). |
| `--simple`         | Skip heavy statistics (faster, less detail).               |
| `-v` / `--verbose` | Show progress messages and any warnings on the console.    |

------

## Output format

The profiler writes a single JSON document. Below is a trimmed example that illustrates the schema:

```json
{
  "file": "sample.csv",
  "shape": { "rows": 1000, "columns": 6 },
  "columns": [
    {
      "name": "customer_id",
      "type": "numeric",
      "total_values": 1000,
      "missing": 0,
      "samples": ["1", "2", "3"],
      "analysis": "numeric",
      "statistics": {
        "min": 1,
        "max": 1000,
        "mean": 500.5,
        "median": 500,
        "std": 288.7
      },
      "zero_count": 0,
      "negative_count": 0,
      "pattern": "likely_id"
    },
    {
      "name": "revenue",
      "type": "currency",
      "total_values": 995,
      "missing": 5,
      "samples": [" $1,250.50 ", " $(890.25)", " $2,100.00 "],
      "analysis": "currency",
      "currency_symbol": "$",
      "valid_numbers": 990,
      "invalid_numbers": 5,
      "statistics": {
        "min": 45.0,
        "max": 9876.0,
        "mean": 1425.75,
        "median": 1200.0,
        "std": 850.3
      },
      "raw_statistics": {
        "min": 0.01,
        "max": 98760.0,
        "note": "Raw min/max before 1%-99% outlier clipping"
      },
      "quartiles": { "q25": 750.0, "q75": 2100.0 },
      "zero_count": 0,
      "negative_count": 12
    },
    {
      "name": "signup_date",
      "type": "datetime",
      "total_values": 980,
      "missing": 20,
      "samples": ["2023-01-15", "2023-02-03", "2023-03-22"],
      "analysis": "datetime",
      "date_range": ["2022-01-01", "2024-12-31"],
      "valid_dates": 970,
      "invalid_dates": 10,
      "success_rate": 99.0
    },
    {
      "name": "country",
      "type": "categorical",
      "total_values": 1000,
      "missing": 0,
      "samples": ["US", "DE", "FR"],
      "analysis": "categorical",
      "categories": 12,
      "values": {"US": 400, "DE": 180, "FR": 150, "GB": 120, "CA": 50},
      "most_common_percentage": 40.0
    },
    {
      "name": "active",
      "type": "boolean",
      "total_values": 1000,
      "missing": 0,
      "samples": ["Yes", "No", "Yes"],
      "analysis": "boolean",
      "true_count": 820,
      "false_count": 180,
      "other_count": 0,
      "true_percentage": 82.0
    },
    {
      "name": "notes",
      "type": "text",
      "total_values": 950,
      "missing": 50,
      "samples": ["VIP client", "Follow-up needed", "No response"],
      "analysis": "text",
      "avg_length": 34.2,
      "max_length": 128,
      "min_length": 0,
      "contains_numbers": 120,
      "contains_special_chars": 45
    }
  ],
  "warnings": [],
  "info": [
    "Read with encoding=utf-8-sig, separator=',', engine=c"
  ],
  "metadata": {
    "file_size_bytes": 842361,
    "file_size_mb": 0.8,
    "memory_usage_mb": 2.3,
    "total_cells": 6000
  }
}
```

### Key fields per column

| Field                | Meaning                                                      |
| -------------------- | ------------------------------------------------------------ |
| `type`               | Detected high-level type (`numeric`, `currency`, `datetime`, `categorical`, `boolean`, `text`). |
| `missing`            | Count of null/missing values.                                |
| `samples`            | Three representative non-null values (truncated to 100 chars). |
| `analysis`           | Sub-section name indicating which block of stats follows.    |
| Numeric-specific     | `statistics` (clipped 1%-99%), `raw_statistics` (true min/max, only present when outliers exist), `pattern` (`likely_id` or `likely_categorical`). |
| Currency-specific    | Same as numeric plus `currency_symbol` (detected symbol, e.g. `$`, `€`, `£`, `¥`). Handles comma-separated values, accounting-style negatives `(1,234.56)`, and all Unicode currency symbols. |
| Datetime-specific    | `date_range`, `valid_dates`, `invalid_dates`, `success_rate`. |
| Categorical-specific | `categories` (distinct count), `values` (top 20 with counts), `most_common_percentage`. |
| Boolean-specific     | `true_count`, `false_count`, `other_count`, `true_percentage`. |
| Text-specific        | Length stats, presence of numbers/special characters.        |

### `warnings` vs `info`

The output separates two distinct message types:

| Field      | Contains                                                     |
| ---------- | ------------------------------------------------------------ |
| `warnings` | Genuine data quality or file issues (e.g. large file, fallback parsing used, all read methods failed). |
| `info`     | Informational messages about how the file was successfully read (encoding, separator, and engine used). |

A clean run will have an empty `warnings` array and one entry in `info`.

------

## Technical notes

| Aspect                  | Detail                                                       |
| ----------------------- | ------------------------------------------------------------ |
| **Framework**           | Tkinter (cross-platform GUI). Uses `trace_add()` for Tcl 9 / Python 3.14+ compatibility. |
| **Data engine**         | pandas for CSV parsing and statistical calculations.         |
| **Threading**           | GUI launches a background `threading.Thread`; results are passed back via a `queue.Queue`. |
| **Encoding strategy**   | Tries `utf-8-sig` first (handles BOM files cleanly), then `utf-8`, `cp1252`, `iso-8859-1`, `latin1`, `ascii`, `utf-16`, `utf-32`. Each encoding is tried with both the C engine (fast, strict) and Python engine (slower, more permissive) before moving on. |
| **Currency detection**  | Uses compiled regex against the Unicode currency symbols block (`U+20A0`-`U+20CF`) plus `$`, `£`, `¥`, `€`. Requires 80% of sampled values to match the pattern and at least one symbol present to avoid false positives. Accounting-style negatives like `$(1,234.56)` are converted to `-1234.56`. |
| **Numeric outliers**    | `statistics` values (min, max, mean, std) are clipped to the 1%-99% range. When the true min/max differ, they are preserved separately under `raw_statistics`. |
| **Large-file handling** | Files over 500 MB trigger a warning in the `warnings` array. |
| **Memory usage**        | `df.memory_usage(deep=True)` is reported in `metadata`.      |
| **Error resilience**    | Every public method catches exceptions and returns a sensible default so the program never crashes. |

------

## CSV reading strategy

The profiler attempts to read each file in four stages, stopping as soon as one succeeds:

1. **Strategy 1** — Iterates all encoding + separator combinations, trying both the C engine (fast, strict) and Python engine (slower, more permissive) for each. This is the primary path and handles the vast majority of files including BOM-encoded CSVs.
2. **Strategy 1.5** — Tries common encodings with pandas auto-separator detection. Catches edge cases that slip past Strategy 1.
3. **Strategy 2** — UTF-8 with full auto-separator fallback. If this is reached, a warning is added to the output.
4. **Strategy 3** — Manual line-by-line text parsing as a last resort.

A successful read is always recorded in the `info` field, not `warnings`. The info message includes the encoding, separator, and engine that succeeded (e.g. `"Read with encoding=utf-8-sig, separator=',', engine=c"`).

------

## Error handling

| Situation                   | Message shown to the user                                    |
| --------------------------- | ------------------------------------------------------------ |
| File not found / unreadable | "Error: `<path>` not found or cannot be opened."             |
| Empty file                  | "File is empty - nothing to profile."                        |
| Unsupported encoding        | "All encoding attempts failed - please verify the file's character set." |
| Very large file (> 500 MB)  | Warning added to JSON `warnings` array and shown in the GUI log. |
| Unexpected parsing error    | "Critical analysis failure: `<exception>` - see log for stack trace." |

All messages are logged to the GUI's text pane and printed to `stderr` in CLI mode.

------

## Typical use cases

- **Quick data discovery** - Open an unknown CSV and get a concise summary within seconds.
- **AI-assistant pipelines** - Feed the JSON profile to LLMs that need structural hints before prompting.
- **Data-quality audits** - Spot missing values, outliers, and inconsistent types early.
- **Pre-processing planning** - Decide which columns need cleaning, encoding, or transformation before feeding data to a model.
- **Documentation generation** - Export the JSON as part of a dataset's metadata bundle.

------

## License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

You are free to:
- Use, share, and adapt this work
- Use it at your job

Under these terms:
- **Attribution** — Credit the original author
- **NonCommercial** — No selling or commercial products
- **ShareAlike** — Derivatives must use the same license