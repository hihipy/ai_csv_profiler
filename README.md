# ai-csv-profiler

A Python application that generates essential CSV data profiles optimized for AI assistant consumption. Features both GUI and command-line interfaces for quick data analysis with minimal, actionable insights.

## Features

- **AI-Optimized Output**: Generates concise JSON profiles with just the essential information AI assistants need

- Dual Interface

  :

  - Modern GUI built with Tkinter for interactive use
  - Command-line interface for automation and scripting

- **Smart Type Detection**: Automatically identifies numeric, text, date, and boolean data types

- Advanced Numeric Analysis

  :

  - Statistical summaries (mean, median, quartiles, standard deviation)
  - Outlier detection using IQR method
  - Distribution skewness analysis
  - Automatic ID and categorical detection
  - Currency/financial data recognition

- Robust File Handling

  :

  - Multiple encoding support (UTF-8, CP1252, ISO-8859-1)
  - Large file warnings (>100MB)
  - File validation and format checking

- **Thread-Safe GUI**: Non-blocking analysis with progress indicators

- **Comprehensive Error Handling**: User-friendly error messages and validation

## Requirements

Ensure Python 3.6+ is installed, then run the following to install required libraries:

```bash
pip install pandas
```

### Linux Users

If you encounter a `tkinter` import error, install it using your distribution's package manager:

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# CentOS/RHEL/Fedora
sudo yum install tkinter
# or
sudo dnf install python3-tkinter
```

**Note**: The following libraries are included with Python's standard library and don't need separate installation:

- `tkinter` (GUI framework)
- `threading`, `queue`, `argparse`, `json`, `os`, `pathlib` (built-in modules)

## Installation

1. Download the script:

```bash
curl -O https://raw.githubusercontent.com/your-repo/ai-csv-profiler/main/ai_csv_profiler.py
```

1. Make it executable (optional):

```bash
chmod +x ai_csv_profiler.py
```

## Usage

### GUI Mode (Default)

Start the graphical interface by running the script without arguments:

```bash
python ai_csv_profiler.py
```

1. **Select File**: Click "Browse..." to choose your CSV file
2. **Choose Output**: Click "Save As..." to specify where to save the JSON profile
3. **Analyze**: Click "🔍 Analyze CSV" to generate the profile
4. **View Results**: The profile is saved as JSON and a summary is displayed

### Command Line Mode

Analyze CSV files directly from the command line:

```bash
# Basic usage - output to console
python ai_csv_profiler.py data.csv

# Save to specific file
python ai_csv_profiler.py data.csv -o profile.json

# Using output flag
python ai_csv_profiler.py data.csv --output analysis_results.json
```

## Output Format

The profiler generates a JSON file with the following structure:

```json
{
  "file": "sample.csv",
  "shape": {
    "rows": 1000,
    "columns": 5
  },
  "columns": [
    {
      "name": "customer_id",
      "type": "number",
      "missing": 0,
      "unique": 1000,
      "samples": ["1", "2", "3"],
      "likely_id": true,
      "range": [1, 1000]
    },
    {
      "name": "revenue",
      "type": "number",
      "missing": 5,
      "unique": 847,
      "samples": ["1250.50", "890.25", "2100.00"],
      "likely_currency": true,
      "mean": 1425.75,
      "median": 1200.00,
      "outlier_count": 23,
      "distribution": "right_skewed"
    }
  ]
}
```

## Column Analysis Features

### All Columns

- **Name**: Column header
- **Type**: Simplified type (number, text, date, boolean)
- **Missing**: Count of null/empty values
- **Unique**: Number of unique values
- **Samples**: 3 sample values for context

### Numeric Columns

- **Statistical Summary**: Mean, median, standard deviation, quartiles

- **Range**: Minimum and maximum values

- **Zero/Negative Counts**: Special value analysis

- **Outlier Detection**: Count and percentage of outliers using IQR method

- **Distribution Analysis**: Skewness detection (normal, left/right skewed)

- Smart Classification

  :

  - **ID Detection**: Sequential unique numbers
  - **Categorical Detection**: Low cardinality integers
  - **Currency Detection**: Financial keyword recognition

### Categorical Columns

- **Value Counts**: Frequency distribution for columns with ≤50 unique values

## Technical Details

- **Framework**: Built using Python's Tkinter for cross-platform GUI compatibility
- **Data Processing**: Powered by pandas for robust CSV parsing and analysis
- **Multi-Threading**: Background processing ensures responsive GUI during analysis
- **Encoding Support**: Automatic fallback through multiple encodings for international data
- **Memory Efficient**: Processes large files without loading entire dataset into memory
- **File Validation**: Pre-analysis checks for file size, format, and readability

## Error Handling

The application includes comprehensive error handling:

- **File Validation**: Checks for existence, readability, and appropriate format
- **Encoding Issues**: Automatic fallback through common encodings
- **Large File Warnings**: User confirmation for files >100MB
- **Malformed Data**: Graceful handling of parsing errors
- **Permission Issues**: Clear error messages for file access problems

## Use Cases

- **Data Exploration**: Quick overview of unknown CSV files
- **AI Assistant Integration**: Generate profiles for ChatGPT, Claude, or other AI tools
- **Data Quality Assessment**: Identify missing values, outliers, and data types
- **Preprocessing Planning**: Understand data characteristics before analysis
- **Documentation**: Create structured metadata for datasets

## Command Line Examples

```bash
# Analyze sales data
python ai_csv_profiler.py sales_2024.csv -o sales_profile.json

# Quick console output for small files
python ai_csv_profiler.py customer_data.csv

# Batch processing (shell script)
for file in *.csv; do
    python ai_csv_profiler.py "$file" -o "${file%.csv}_profile.json"
done
```

## Acknowledgments

- [Pandas](https://pandas.pydata.org/) for powerful data analysis capabilities
- [Python's Tkinter](https://docs.python.org/3/library/tkinter.html) for cross-platform GUI development
- [JSON](https://www.json.org/) specification for structured data exchange

------

ai-csv-profiler © 2025 is licensed under [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/)