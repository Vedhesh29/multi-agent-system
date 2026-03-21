from graph import build_graph, build_merged_graph, build_controller_graph, build_compressed_graph, build_optimized_graph, build_revision_graph
from agents import score_output
import json
import csv, os

app = build_revision_graph()

tasks = [
    "Explain the causes of the 2008 financial crisis in 3 paragraphs.",
    "Explain how transformers work in NLP.",
    "Summarize the pros and cons of renewable energy.",
    "Describe the key events of World War 1.",
    "Explain gradient descent in machine learning.",
    "What are the main causes of inflation?"
]

def log_run(task, result, variant="baseline"):
    quality_score = score_output(task, result["final_output"])
    total_tokens = sum(result["token_usage"].values())
    row = {
        "variant": variant,
        "task": task,
        "total_latency": sum(result["latency"].values()),
        "total_tokens": total_tokens,
        "quality_score": quality_score,
        **{f"latency_{k}": v for k, v in result["latency"].items()},
        **{f"tokens_{k}": v for k, v in result["token_usage"].items()}
    }
    filename = f"{variant}_metrics.csv"
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

for task in tasks:
    print(f"\n=== RUNNING TASK: {task} ===\n")
    initial_state = {
        "task": task,
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
    print("\n=== TOKEN USAGE ===")
    print(json.dumps(result["token_usage"], indent=2))
    log_run(task, result, variant="revision")