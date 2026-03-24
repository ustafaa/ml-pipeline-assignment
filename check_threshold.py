import os
import sys

THRESHOLD = 0.85


def read_model_info(path="model_info.txt"):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.", file=sys.stderr)
        sys.exit(1)
    content = open(path).read().strip()
    if not content or "," not in content:
        print(f"ERROR: {path} is invalid. Expected 'run_id,accuracy'.", file=sys.stderr)
        sys.exit(1)
    run_id, accuracy = content.split(",", 1)
    return run_id, float(accuracy)


def check_threshold():
    run_id, accuracy = read_model_info()
    if accuracy < THRESHOLD:
        print(f"FAIL: accuracy {accuracy:.4f} is below threshold {THRESHOLD}")
        sys.exit(1)
    print(f"PASS: accuracy {accuracy:.4f} meets threshold {THRESHOLD}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    check_threshold()
