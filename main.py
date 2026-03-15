from graph import build_graph
import json
import csv, os

app = build_graph()

initial_state = {
    "task": "Explain the causes of the 2008 financial crisis in 3 paragraphs.",
    "plan": None,
    "research": None,
    "draft": None,
    "critique": None,
    "final_output": None,
    "token_usage": {},
    "latency": {}
}

result = app.invoke(initial_state)

print("=== FINAL OUTPUT ===")
print(result["final_output"])
print("\n=== CRITIQUE ===")
print(result["critique"])
print("\n=== LATENCY PER AGENT ===")
print(json.dumps(result["latency"], indent=2))

def log_run(task, result, variant="baseline"):
    row = {
        "variant": variant,
        "task": task,
        "total_latency": sum(result["latency"].values()),
        **{f"latency_{k}": v for k, v in result["latency"].items()}
    }
    file_exists = os.path.exists("metrics.csv")
    with open("metrics.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

log_run(initial_state["task"], result)