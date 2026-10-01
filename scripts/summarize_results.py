"""Print committed PushT measurements without model weights or ML dependencies."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    baseline = json.loads((root / "ddpm_unet_s42/eval_results.json").read_text())
    print("Historical measurements; legacy success_rate is intentionally excluded.")
    print("experiment,score,max_coverage,p50_ms,speedup_vs_ddpm")
    for path in sorted(root.glob("*/eval_results.json")):
        row = json.loads(path.read_text())
        latency = row["inference_latency_p50_ms"]
        speedup = baseline["inference_latency_p50_ms"] / latency
        print(f"{path.parent.name},{row['test_mean_score']:.4f},"
              f"{row['test_target_area_coverage']:.4f},{latency:.2f},{speedup:.2f}")


if __name__ == "__main__":
    main()
