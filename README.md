# ai-csv-profiler

A Python utility that builds concise, AI-friendly JSON profiles of CSV files. Includes both a Tkinter GUI and a command-line interface, so you can explore data interactively or automate profiling in scripts.

------

## What the tool does

| Feature                          | Why it matters                                               |
| -------------------------------- | ------------------------------------------------------------ |
| **AI-Optimized JSON output**     | Gives downstream assistants (ChatGPT, Claude, etc.) exactly the fields they need. |
| **Dual UI: GUI + CLI**           | Pick the workflow that fits: point-and-click or batch processing. |
| **Smart type detection**         | Auto-recognizes numeric, text, date/time, boolean, and categorical columns. |
| **Rich numeric analysis**        | Means, medians, quartiles, IQR-based outlier count, skewness, auto-ID/currency detection. |
| **Robust file handling**         | Tries several encodings, warns on huge files, validates format before parsing. |
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

> **Note:** `tkinter`, `threading`, `queue`, `argparse`, `json`, `os`, `pathlib` are part of the Python standard library and need no extra installation.

------

## Installation

```bash
# Grab the latest script
curl -O https://raw.githubusercontent.com/your-repo/ai-csv-profiler/main/ai_csv_profiler.py

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
  "shape": { "rows": 1000, "columns": 5 },
  "columns": [
    {
      "name": "customer_id",
      "type": "numeric",
      "non_null": 1000,
      "nulls": 0,
      "samples": ["1", "2", "3"],
      "analysis": "numeric",
      "min": 1,
      "max": 1000,
      "mean": 500.5,
      "median": 500,
      "std": 288.7,
      "zeroes": 0,
      "positives": 1000,
      "negatives": 0,
      "distinct": 1000,
      "likely_id": true
    },
    {
      "name": "revenue",
      "type": "numeric",
      "non_null": 995,
      "nulls": 5,
      "samples": ["1250.50", "890.25", "2100.00"],
      "analysis": "numeric",
      "min": 45.0,
      "max": 9876.0,
      "mean": 1425.75,
      "median": 1200.0,
      "std": 850.3,
      "outlier_count": 23,
      "distribution": "right_skewed",
      "likely_currency": true
    },
    {
      "name": "signup_date",
      "type": "datetime",
      "non_null": 980,
      "nulls": 20,
      "samples": ["2023-01-15", "2023-02-03", "2023-03-22"],
      "analysis": "datetime",
      "min": "2022-01-01",
      "max": "2024-12-31",
      "distinct": 970
    },
    {
      "name": "country",
      "type": "categorical",
      "non_null": 1000,
      "nulls": 0,
      "samples": ["US", "DE", "FR"],
      "analysis": "categorical",
      "distinct": 12,
      "top_five": {"US": 400, "DE": 180, "FR": 150, "GB": 120, "CA": 50},
      "mode": "US"
    },
    {
      "name": "notes",
      "type": "text",
      "non_null": 950,
      "nulls": 50,
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
  "metadata": {
    "size_bytes": 842361,
    "size_mb": 0.8,
    "memory_usage_mb": 2.3,
    "column_types": {
      "customer_id": "object",
      "revenue": "object",
      "signup_date": "object",
      "country": "object",
      "notes": "object"
    }
  }
}
```

### Key fields per column

| Field                | Meaning                                                      |
| -------------------- | ------------------------------------------------------------ |
| `type`               | Detected high-level type (`numeric`, `datetime`, `categorical`, `text`). |
| `non_null` / `nulls` | Count of present vs. missing values.                         |
| `samples`            | Three representative non-null values (truncated to 100 chars). |
| `analysis`           | Sub-section name indicating which block of stats follows.    |
| Numeric-specific     | `min`, `max`, `mean`, `median`, `std`, `outlier_count`, `distribution`, `likely_id`, `likely_currency`. |
| Datetime-specific    | `min`, `max`, `distinct`, `nulls_after_parse`.               |
| Categorical-specific | `distinct`, `top_five` (value to count), `mode`.             |
| Text-specific        | Length stats, presence of numbers/special characters.        |

------

## Technical notes

| Aspect                  | Detail                                                       |
| ----------------------- | ------------------------------------------------------------ |
| **Framework**           | Tkinter (cross-platform GUI).                                |
| **Data engine**         | pandas for CSV parsing and statistical calculations.         |
| **Threading**           | GUI launches a background `threading.Thread`; results are passed back via a `queue.Queue`. |
| **Encoding fallback**   | Tries UTF-8, UTF-8-sig, CP1252, ISO-8859-1, latin1, ASCII, UTF-16, UTF-32. |
| **Large-file handling** | Files over 500 MB are read in chunks; a warning is added to the JSON `warnings` array. |
| **Memory usage**        | `df.memory_usage(deep=True)` is reported; the profiler avoids loading the whole file into RAM when it can be chunked. |
| **Error resilience**    | Every public method catches exceptions and returns a sensible default, so the program never crashes. |

------

## Error handling

| Situation                   | Message shown to the user                                    |
| --------------------------- | ------------------------------------------------------------ |
| File not found / unreadable | "Error: `<path>` not found or cannot be opened."             |
| Empty file                  | "File is empty - nothing to profile."                        |
| Unsupported encoding        | "All encoding attempts failed - please verify the file's character set." |
| Very large file (> 100 MB)  | "Large file detected (approx X MB). Continue? (Y/N)" (CLI) / GUI shows a modal warning. |
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

ai-csv-profiler © 2025

Distributed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.

------

## Acknowledgments

- **pandas** - Data-frame abstraction and statistical calculations.
- **Tkinter** - Lightweight, cross-platform GUI without external dependencies.

------

**Tip:** If you need the latest fixes or want to contribute, fork the repository, push your changes, and open a Pull Request. The code is modular, so adding new column analyses (e.g., sentiment scoring for text) is straightforward.
