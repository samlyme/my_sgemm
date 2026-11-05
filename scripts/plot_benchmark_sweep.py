# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8",
# ]
# ///

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter


@dataclass(frozen=True)
class Result:
    kernel: str
    size: int
    iterations: int
    time_ms: float
    gflops: float
    measured_at: str


FAMILIES = (
    ("cuBLAS", lambda kernel: kernel == "cublas"),
    ("Naive", lambda kernel: kernel.startswith("naive")),
    ("Basic tiled", lambda kernel: kernel.startswith("tiled-basic")),
    (
        "General tiled",
        lambda kernel: kernel.startswith("tiled-general-")
        and not kernel.startswith("tiled-general-coarsened"),
    ),
    (
        "Coarsened tiled",
        lambda kernel: kernel.startswith("tiled-general-coarsened-")
        and not kernel.startswith("tiled-general-coarsened-register"),
    ),
    (
        "Register-staged",
        lambda kernel: kernel.startswith("tiled-general-coarsened-register"),
    ),
)

COLORS = {
    "cuBLAS": "#111111",
    "Naive": "#d55e00",
    "Basic tiled": "#e69f00",
    "General tiled": "#56b4e9",
    "Coarsened tiled": "#009e73",
    "Register-staged": "#0072b2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the best SGEMM configuration in each kernel family."
    )
    parser.add_argument(
        "--input", default="benchmarks/sweep.csv", help="benchmark CSV"
    )
    parser.add_argument(
        "--output-dir", default="benchmarks/plots", help="plot output directory"
    )
    return parser.parse_args()


def family_for(kernel: str) -> str | None:
    for name, matches in FAMILIES:
        if matches(kernel):
            return name
    return None


def read_latest_results(path: Path) -> list[Result]:
    latest: dict[tuple[str, int], Result] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["check"] != "PASS":
                continue
            result = Result(
                kernel=row["kernel"],
                size=int(row["size"]),
                iterations=int(row["iterations"]),
                time_ms=float(row["time_ms"]),
                gflops=float(row["gflops"]),
                measured_at=row["measured_at_utc"],
            )
            key = (result.kernel, result.size)
            previous = latest.get(key)
            if previous is None or result.measured_at >= previous.measured_at:
                latest[key] = result
    return list(latest.values())


def select_best_by_family(results: list[Result]) -> dict[str, list[Result]]:
    candidates: dict[tuple[str, int], list[Result]] = defaultdict(list)
    for result in results:
        family = family_for(result.kernel)
        if family is not None:
            candidates[(family, result.size)].append(result)

    selected: dict[str, list[Result]] = defaultdict(list)
    for (family, _), family_results in candidates.items():
        selected[family].append(max(family_results, key=lambda result: result.gflops))
    for family_results in selected.values():
        family_results.sort(key=lambda result: result.size)
    return selected


def write_summary(path: Path, selected: dict[str, list[Result]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            ["family", "size", "kernel", "iterations", "time_ms", "gflops"]
        )
        for family, _ in FAMILIES:
            for result in selected.get(family, []):
                writer.writerow(
                    [
                        family,
                        result.size,
                        result.kernel,
                        result.iterations,
                        f"{result.time_ms:.6f}",
                        f"{result.gflops:.3f}",
                    ]
                )


def configure_axes(axis, sizes: list[int], ylabel: str) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(sizes)
    axis.get_xaxis().set_major_formatter(ScalarFormatter())
    axis.get_xaxis().set_minor_formatter(NullFormatter())
    axis.tick_params(axis="x", which="minor", bottom=False)
    axis.set_xlabel("Square matrix dimension (M = N = K)")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.45)


def plot_metric(
    selected: dict[str, list[Result]],
    sizes: list[int],
    output_dir: Path,
    metric: str,
) -> None:
    if metric == "gflops":
        title = "SGEMM throughput: best configuration in each family"
        ylabel = "Throughput (GFLOP/s)"
        filename = "throughput_by_family"
    else:
        title = "SGEMM latency: best configuration in each family"
        ylabel = "Average kernel latency (ms)"
        filename = "latency_by_family"

    figure, axis = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    for family, _ in FAMILIES:
        family_results = selected.get(family, [])
        if not family_results:
            continue
        axis.plot(
            [result.size for result in family_results],
            [getattr(result, metric) for result in family_results],
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=family,
            color=COLORS[family],
        )

    configure_axes(axis, sizes, ylabel)
    if metric == "time_ms":
        axis.set_yscale("log")
    else:
        axis.set_ylim(bottom=0)
    axis.set_title(title)
    axis.legend(loc="best", frameon=True)

    for extension in ("png", "svg"):
        output_path = output_dir / f"{filename}.{extension}"
        figure.savefig(output_path, dpi=180)
        if extension == "svg":
            lines = output_path.read_text().splitlines()
            output_path.write_text("\n".join(line.rstrip() for line in lines) + "\n")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = read_latest_results(input_path)
    selected = select_best_by_family(results)
    sizes = sorted({result.size for values in selected.values() for result in values})
    if not sizes:
        raise SystemExit(f"no passing benchmark rows found in {input_path}")

    write_summary(output_dir / "best_by_family.csv", selected)
    plot_metric(selected, sizes, output_dir, "gflops")
    plot_metric(selected, sizes, output_dir, "time_ms")
    print(f"wrote benchmark plots to {output_dir}")


if __name__ == "__main__":
    main()
