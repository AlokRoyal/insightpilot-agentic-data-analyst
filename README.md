# InsightPilot | Agentic Data Analyst

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

**Explore public data, ask questions in plain language, and get answers grounded in reproducible analytics.**

InsightPilot is an agentic analytics portfolio project. When a model is available, it plans an analysis by selecting from explicit tools; Python and pandas calculate the numbers. The model explains the results with coverage notes and caveats. The chat workflow never executes model-written Python. If neither OpenAI nor local Ollama is configured, common questions still work through a deterministic question router.

The dashboard follows five World Bank indicators across eight countries. The dataset refreshes weekly. A separate daily code agent can propose one small analytics extension as a **draft pull request**; it cannot merge or push generated code to `main`.

## What it demonstrates

- **Agentic tool use:** the model can plan a multi-step answer by calling summary, trend, country comparison, and correlation tools.
- **Grounded analysis:** numeric results come from deterministic pandas functions, not language-model arithmetic.
- **Data engineering:** a public REST API becomes a validated, versioned, tidy CSV snapshot.
- **Extensible analytics:** insight modules share a `run(frame) -> dict` interface and are loaded independently.
- **Guardrailed code evolution:** a daily enhancement queue creates isolated modules, validates Python syntax and opens a draft PR for human review.
- **Transparent limitations:** missing values remain missing, indicator years can differ, and correlations are not presented as causal effects.

## Try it

Example questions supported by the built-in analysis tools:

- How has GDP per capita in India changed since 2000?
- Compare internet use across India, Bangladesh, and Indonesia in the latest shared year.
- Is internet use associated with GDP per capita across countries? What are the limitations?

The dashboard displays the observation count, country and indicator coverage, a selectable time-series chart, a data preview, caveats, and a data-quality summary.

## Architecture

```text
World Bank API --weekly--> validated CSV --> Streamlit dashboard
                                                |-- Plotly exploration
User question --> OpenAI or local Ollama --> bounded tool calls --> pandas results --> explanation
                                                |-- pluggable insights
Daily schedule --> enhancement queue --> isolated extension --> syntax check --> draft PR
```

## Start locally

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # optional; configure a model provider if available
streamlit run app.py
```

The first run fetches the public World Bank data using five parallel multi-country requests. The validated CSV is cached locally for subsequent starts. Charts, snapshots, and deterministic insight modules need no model. For natural-language analysis, the app uses OpenAI when `OPENAI_API_KEY` is set; without one, it connects to a local Ollama model when available and falls back to deterministic tools for common trend, comparison, summary, and correlation questions. Keep `.env` private. OpenAI API use may incur charges under your provider account.

### Run the GenAI analyst without an API key

1. Install Ollama for Windows from [ollama.com/download/windows](https://ollama.com/download/windows) (Windows 10 or later).
2. In PowerShell, download a tool-calling model: `ollama pull qwen3:4b`.
3. Keep Ollama running, then start InsightPilot with `streamlit run app.py`.

The app sends prompts and tool results to the Ollama service on your computer at `http://localhost:11434`. Change `OLLAMA_BASE_URL` or `OLLAMA_MODEL` in `.env` to use another local Ollama model. The model chooses among the same bounded pandas analysis tools; it cannot run arbitrary Python. Without a model, the deterministic router supports common summaries, trends, comparisons, and correlations and labels its answers accordingly.

To manually refresh the data, click **Refresh data now** in the sidebar or run:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from insight_pilot.data import refresh_data; print(refresh_data())"
```

You can also upload a CSV with `country_code,country,year,indicator,indicator_code,value` columns.

## Schedules

GitHub Actions schedules use UTC and can also be started with **Run workflow**:

| Workflow | Schedule | Result |
|---|---|---|
| Weekly public data refresh | Monday, 02:00 UTC (07:30 IST) | Fetches the latest available World Bank observations and commits the CSV only when it changes |
| Daily agentic code evolution | Every day, 02:30 UTC (08:00 IST) | Opens a draft PR for one queued analytics extension after validation |

GitHub Actions can delay scheduled runs during busy periods. These are best-effort schedules, not guaranteed exact-time alerts.

### Enable the daily code agent

1. The repository Actions permission to create pull requests must be enabled. The workflow token is scoped to contents and pull requests.
2. The workflow reads `config/daily_enhancements.json`. It opens a draft PR for one new timestamped extension under `src/insight_pilot/extensions/` each day.
3. With an `OPENAI_API_KEY` secret, the extension is AI-generated. Without it, the job uses a curated deterministic implementation from `daily_features.py`; daily code proposals still run without paid API access.
4. Review each module before merging. The workflow validates syntax and restricts imports/calls. It does not merge PRs or modify other files on `main`.

The daily hosted workflow cannot use a local model running on your PC. Its no-key fallback uses the checked-in feature library, while the interactive app can use your local Ollama model. The weekly public-data workflow also needs no API key.

## Data and analysis details

**Source:** [World Bank Indicators API](https://api.worldbank.org/). The project fetches GDP per capita, total population, life expectancy, internet usage, and unemployment for India, Bangladesh, Brazil, China, Germany, Indonesia, Japan, and the United States. The API's latest available year varies by measure and country.

The agent exposes four bounded functions: summarize an indicator, calculate a country trend, compare countries in one year, and calculate a descriptive Pearson correlation. Invalid indicator or country names return clear errors. Tool calls are capped to prevent unbounded loops. Insight plug-ins are separate Python modules with one `run(frame)` function.

## Project layout

```text
app.py                              Streamlit dashboard
src/insight_pilot/agent.py          OpenAI/Ollama agent and deterministic fallback
src/insight_pilot/analytics.py      Deterministic analytics tools
src/insight_pilot/data.py           Parallel World Bank fetch and validation
src/insight_pilot/extensions/       Pluggable insights + no-key feature library
scripts/daily_agent.py              Bounded code-proposal generator
config/daily_enhancements.json      Daily feature queue
.github/workflows/                  Weekly data + daily code PR schedules
```

## Roadmap

- Expand the insight plug-in library with robust change-point, peer-group, and coverage analyses.
- Add configurable datasets and user-defined measures.
- Add a downloadable, cited analysis report.
- Add data freshness and run-history panels.

## License

MIT. See [LICENSE](LICENSE).
