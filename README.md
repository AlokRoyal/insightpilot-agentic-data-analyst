# InsightPilot

### Agentic data analysis with bounded tools, public data, and reviewable daily evolution

InsightPilot is a portfolio project that turns natural-language questions into reproducible analyses. A language model selects from a small set of explicit analytics tools; Python and pandas calculate the numbers. The model explains those results with coverage notes and caveats. It never executes model-written Python in the chat workflow.

The dashboard follows five World Bank indicators across eight countries. The dataset refreshes weekly. A separate daily code agent can propose one small analytics extension as a **draft pull request**; it cannot merge or push generated code to `main`.

## What it demonstrates

- **Agentic tool use:** the model can plan a multi-step answer by calling summary, trend, country comparison, and correlation tools.
- **Grounded analysis:** numeric results come from deterministic pandas functions, not language-model arithmetic.
- **Data engineering:** a public REST API becomes a validated, versioned, tidy CSV snapshot.
- **Extensible analytics:** insight modules share a `run(frame) -> dict` interface and are loaded independently.
- **Guardrailed code evolution:** a daily enhancement queue creates isolated modules, validates Python syntax and opens a draft PR for human review.
- **Transparent limitations:** missing values remain missing, indicator years can differ, and correlations are not presented as causal effects.

## Architecture

```text
World Bank API --weekly--> validated CSV --> Streamlit dashboard
                                                |-- Plotly exploration
User question --> OpenAI Responses API --> bounded tool calls --> pandas results --> explanation
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
copy .env.example .env  # PowerShell; edit .env and add your own API key
streamlit run app.py
```

The first run fetches the public World Bank data. The OpenAI API key is only needed for the natural-language agent; charts, snapshots, and deterministic insight modules work without it. Keep `.env` private. API use may incur charges under your provider account.

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
| Daily agentic code evolution | Every day, 02:30 UTC (08:00 IST) | Generates one queued extension and opens a draft PR after a syntax check |

GitHub Actions can delay scheduled runs during busy periods. These are best-effort schedules, not guaranteed exact-time alerts.

### Enable the daily code agent

1. Add a repository Actions secret named `OPENAI_API_KEY` with an API key you control. Never put the key in source code or commit it.
2. In repository **Settings → Actions → General → Workflow permissions**, allow GitHub Actions to create pull requests. The workflow's token is scoped to contents and pull requests.
3. The workflow reads `config/daily_enhancements.json`. It proposes one new file under `src/insight_pilot/extensions/` and opens a draft pull request.
4. Review each generated module before merging. Daily modules are timestamped and loaded as isolated plug-ins. The workflow validates syntax and restricts imports/calls, but generated code still needs human review. It does not merge PRs or modify other files.

Without the secret or the repository permission, the scheduled code proposal will not complete. The weekly public-data workflow does not require an API key.

## Data and analysis details

**Source:** [World Bank Indicators API](https://api.worldbank.org/). The project fetches GDP per capita, total population, life expectancy, internet usage, and unemployment for India, Bangladesh, Brazil, China, Germany, Indonesia, Japan, and the United States. The API's latest available year varies by measure and country.

The agent exposes four bounded functions: summarize an indicator, calculate a country trend, compare countries in one year, and calculate a descriptive Pearson correlation. Invalid indicator or country names return clear errors. Tool calls are capped to prevent unbounded loops. Insight plug-ins are separate Python modules with one `run(frame)` function.

## Project layout

```text
app.py                              Streamlit interface
src/insight_pilot/agent.py          OpenAI tool-calling loop
src/insight_pilot/analytics.py      Deterministic analytics tools
src/insight_pilot/data.py           World Bank fetch and validation
src/insight_pilot/extensions/       Pluggable deterministic insights
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
