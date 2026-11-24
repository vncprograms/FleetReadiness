# FleetReadiness

A practice tool that plays saved jet-engine runs, guesses how many runs are left, and writes a short maintenance note.

The engines are NASA’s C-MAPSS simulation, subset FD001. Every run is at sea level. The screen replays those saved runs. The notes are for the demo.

![Fleet overview: 200 engines, average guess 88.2 runs left, 29 engines that need attention](demo/screenshots/01-fleet-overview.png)

## Instruction manual

Each link opens that part of the manual.

| | |
|---|---|
| [What this is](docs/manual.md#what-this-is) | The practice tool, in one page. |
| [Why it is useful](docs/manual.md#why-it-is-useful) | Which engine to look at before it fails. |
| [What a cycle is](docs/manual.md#what-a-cycle-is) | One run, saved as one row. |
| [The engine files](docs/manual.md#the-engine-files) | 100 engines run to failure, 100 stopped early. |
| [What the model is allowed to see](docs/manual.md#what-the-model-is-allowed-to-see) | 93 numbers, and only past runs. |
| [How many runs are left](docs/manual.md#how-many-runs-are-left) | The three models, and the final-exam misses. |
| [Anomaly scores](docs/manual.md#anomaly-scores) | Higher is more unusual. The alarm line is 0. |
| [The four made-up changes](docs/manual.md#the-four-made-up-changes) | Temperature, pressure, a spike, and a wiggle. |
| [The maintenance note](docs/manual.md#the-maintenance-note) | Low, Medium, and High. |
| [Using it with other things](docs/manual.md#using-it-with-other-things) | A schedule, parts, a chat alert, another machine. |
| [Using the screen](docs/manual.md#using-the-screen) | Links into the click-through. |
| [Limits](docs/manual.md#limits) | What this copy does not do. |

The whole manual is one page: [Instruction manual](docs/manual.md).

The click-through, with its own contents, is [How to use FleetReadiness](HOW_TO_USE.md). Dataset source and the full score tables are in the [technical notes](docs/technical-notes.md). A short recording of the screens is the [demo video](demo/fleetreadiness-demo.webm).

## Run it

Python 3.12. On macOS, XGBoost needs `brew install libomp`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

Then open [http://localhost:8501](http://localhost:8501). The saved models are already in the repo, so the screen works without training again.
