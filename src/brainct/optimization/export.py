"""Export, quantize and certify a model bundle.

    uv run python -m brainct.optimization.export --run artifacts/runs/cnn25d_resnet18 \
        --out artifacts/bundle

1. CNN and series model -> ONNX; parity against PyTorch on real validation series.
2. Static INT8 quantization of the CNN (QDQ, per-channel weights), calibrated on training
   slices. The HU windowing nodes stay FP32: uint8 over the raw HU range would be a 16 HU step,
   a fifth of the brain window.
3. The deployed chain (INT8 CNN -> series model) runs through serving.model.InferenceModel on
   validation and test: calibration and thresholds are fit on validation, test is scored once.
   FP32 ONNX is scored the same way, so the INT8 cost is measured, not assumed.
4. Drift reference histograms from training slices; model card with checksums.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from brainct import __version__
from brainct.constants import IMAGE_SIZE, LABELS, WINDOWS
from brainct.evaluation import calibrate, fit_on_validation, score
from brainct.monitoring.drift import build_reference, slice_features
from brainct.serving.model import InferenceModel, sha256
from brainct.training.data import Series, load_series, read_volume
from brainct.training.model import ExportSliceNet, SeqNet, SliceNet

OPSET = 17


def export_cnn(net: SliceNet, path: Path) -> None:
    x = torch.randn(2, 2 * net.context + 1, IMAGE_SIZE, IMAGE_SIZE) * 200
    torch.onnx.export(
        ExportSliceNet(net),
        (x,),
        str(path),
        dynamo=False,
        opset_version=OPSET,
        input_names=["hu"],
        output_names=["logits", "embedding", "fmap"],
        dynamic_axes={"hu": {0: "n"}, "logits": {0: "n"}, "embedding": {0: "n"}, "fmap": {0: "n"}},
    )


def export_seq(seq: SeqNet, embed_dim: int, path: Path) -> None:
    seq.eval()
    args = (torch.randn(1, 30, embed_dim), torch.randn(1, 30, len(LABELS)))
    torch.onnx.export(
        seq,
        args,
        str(path),
        dynamo=False,
        opset_version=OPSET,
        input_names=["embedding", "logits"],
        output_names=["refined"],
        dynamic_axes={"embedding": {1: "t"}, "logits": {1: "t"}, "refined": {1: "t"}},
    )


def windowing_nodes(model_path: Path) -> list[str]:
    """Names of all nodes upstream of the first Conv (the in-graph HU windowing)."""
    graph = onnx.load(str(model_path)).graph
    producer = {o: n for n in graph.node for o in n.output}
    first_conv = next(n for n in graph.node if n.op_type == "Conv")
    seen: set[str] = set()
    stack = [first_conv.input[0]]
    while stack:
        node = producer.get(stack.pop())
        if node is not None and node.name not in seen:
            seen.add(node.name)
            stack.extend(node.input)
    return sorted(seen)


class SliceReader(CalibrationDataReader):
    def __init__(self, batches: list[np.ndarray]):
        self.it = iter([{"hu": b} for b in batches])

    def get_next(self):
        return next(self.it, None)


def calibration_batches(series: list[Series], context: int, n: int, seed: int) -> list[np.ndarray]:
    """n slices from training series, half from hemorrhage-positive slices."""
    from brainct.serving.model import contexts

    rng = np.random.default_rng(seed)
    pos = [(s, i) for s in range(len(series)) for i in np.nonzero(series[s].labels[:, 0])[0]]
    neg = [(s, i) for s in range(len(series)) for i in np.nonzero(series[s].labels[:, 0] == 0)[0]]
    picks = [pos[k] for k in rng.choice(len(pos), n // 2, replace=False)]
    picks += [neg[k] for k in rng.choice(len(neg), n - n // 2, replace=False)]
    out = []
    for s, i in picks:
        vol = read_volume(series[s])
        out.append(contexts(vol, context)[i : i + 1])
    return [np.concatenate(out[k : k + 16]) for k in range(0, len(out), 16)]


def chain_predictions(model: InferenceModel, series: list[Series]) -> dict[str, dict]:
    """Raw cnn / seq logits of the deployed chain, in the evaluation's npz layout."""
    cnn, seq = [], []
    for s in series:
        out = model.run(read_volume(s), with_seq=True)
        cnn.append(out.cnn_logits)
        assert out.seq_logits is not None
        seq.append(out.seq_logits)
    base = {
        "series_id": np.array([s.series_id for s in series]),
        "patient_id": np.array([s.patient_id for s in series]),
        "offsets": np.concatenate([[0], np.cumsum([s.length for s in series])]),
        "labels": np.concatenate([s.labels for s in series]),
    }
    return {
        "cnn": {**base, "logits": np.concatenate(cnn)},
        "seq": {**base, "logits": np.concatenate(seq)},
    }


def write_card(out: Path, card: dict) -> None:
    card["files"] = {
        name: {"path": path, "sha256": sha256(out / path)}
        for name, path in card.pop("_files").items()
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2))


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/bundle"))
    ap.add_argument("--calib-slices", type=int, default=384)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--max-series", type=int, default=None, help="smoke runs only")
    args = ap.parse_args(argv)
    torch.set_num_threads(8)

    ck = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = SliceNet(cfg["backbone"], cfg["context"], pretrained=False)
    net.load_state_dict(ck["model"])
    net.eval()
    sck = torch.load(args.run / "seq.pt", map_location="cpu", weights_only=False)
    seq = SeqNet(sck["embed_dim"], hidden=sck["hidden"])
    seq.load_state_dict(sck["model"])
    seq.eval()

    work = args.out.with_name(args.out.name + ".work")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    export_cnn(net, work / "cnn.fp32.onnx")
    export_seq(seq, sck["embed_dim"], work / "seq.onnx")
    np.save(work / "cam_weights.npy", net.head.weight.detach().numpy().astype(np.float32))

    data_dir = Path(cfg["data_dir"])
    val = load_series(data_dir, "val")[: args.max_series]
    test = load_series(data_dir, "test")[: args.max_series]
    train = load_series(data_dir, "train")

    # ---- parity: ONNX FP32 vs PyTorch on two real validation series
    import onnxruntime as ort

    from brainct.serving.model import contexts

    sess = ort.InferenceSession(str(work / "cnn.fp32.onnx"), providers=["CPUExecutionProvider"])
    parity = 0.0
    for s in val[:2]:
        x = contexts(read_volume(s), cfg["context"])
        with torch.no_grad():
            ref = net(torch.from_numpy(x)).numpy()
        parity = max(parity, float(np.abs(sess.run(["logits"], {"hu": x})[0] - ref).max()))
    print(f"ONNX FP32 vs PyTorch max |logit diff| = {parity:.2e}")
    assert parity < 1e-3, "ONNX export does not match PyTorch"

    # ---- INT8
    t0 = time.time()
    quant_pre_process(
        str(work / "cnn.fp32.onnx"), str(work / "cnn.prep.onnx"), skip_symbolic_shape=True
    )
    excluded = windowing_nodes(work / "cnn.prep.onnx")
    quantize_static(
        str(work / "cnn.prep.onnx"),
        str(work / "cnn.int8.onnx"),
        SliceReader(calibration_batches(train, cfg["context"], args.calib_slices, seed=7)),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.Percentile,
        nodes_to_exclude=excluded,
        extra_options={"CalibPercentile": 99.99},
    )
    print(f"quantized in {time.time() - t0:.0f} s; FP32 nodes kept: {excluded}")

    # ---- certify both precisions through the serving code path
    results: dict[str, dict[str, Any]] = {}
    chains: dict[str, tuple[Path, dict[str, Any], dict]] = {}
    for precision in ("fp32", "int8"):
        stage = work / f"bundle_{precision}"
        stage.mkdir()
        for f in ("seq.onnx", "cam_weights.npy"):
            shutil.copy(work / f, stage / f)
        shutil.copy(work / f"cnn.{precision}.onnx", stage / "cnn.onnx")
        card: dict[str, Any] = {
            "name": "brainct-ich",
            "version": __version__,
            "contract": {
                "labels": list(LABELS),
                "context": cfg["context"],
                "image_size": IMAGE_SIZE,
                "windows": [list(w) for w in WINDOWS],
                "input": "HU, int16-rounded, padded square + area-resized",
                "batch": 16,
            },
            "calibration": {},
            "thresholds": {},
            "_files": {"cnn": "cnn.onnx", "seq": "seq.onnx", "cam_weights": "cam_weights.npy"},
        }
        write_card(stage, json.loads(json.dumps(card)))
        model = InferenceModel(stage, verify=False)
        t0 = time.time()
        val_pred = chain_predictions(model, val)
        test_pred = chain_predictions(model, test)
        print(f"{precision}: chain on val+test in {time.time() - t0:.0f} s")
        results[precision] = {}
        for head in ("cnn", "seq"):
            fitted = fit_on_validation(val_pred[head])
            results[precision][head] = {
                "fitted": fitted,
                "test": score(test_pred[head], fitted, args.boot),
            }
        chains[precision] = (stage, card, val_pred)

    # ---- final bundle = INT8 chain
    stage, card, val_pred = chains["int8"]
    res = results["int8"]
    card["calibration"] = {h: res[h]["fitted"]["calibration"] for h in ("cnn", "seq")}
    card["thresholds"] = {
        h: {
            **res[h]["fitted"]["thresholds_f1"],
            "screening_any": res[h]["fitted"]["threshold_screening_any"],
            "study_any": res[h]["fitted"]["threshold_study_any"],
        }
        for h in ("cnn", "seq")
    }

    def summary(r: dict) -> dict:
        t = r["test"]
        return {
            "auc_macro": t["auc_macro"],
            "auc_macro_ci": t["auc_macro_ci"],
            "weighted_log_loss": t["weighted_log_loss"],
            "study_any_auc": t["study_any"]["auc"],
            "auc": {k: t["labels"][k]["auc"] for k in LABELS},
        }

    card["metrics_test"] = {p: {h: summary(results[p][h]) for h in ("cnn", "seq")} for p in results}
    card["metrics_test"]["n"] = {
        k: results["int8"]["seq"]["test"][k] for k in ("slices", "series", "patients")
    }

    # drift reference: input features from training slices, p_any from validation (natural prior)
    rng = np.random.default_rng(11)
    feats: dict[str, list[float]] = {}
    for s in [train[i] for i in rng.choice(len(train), min(300, len(train)), replace=False)]:
        vol = read_volume(s)
        for i in range(0, s.length, 3):
            for k, v in slice_features(vol[i]).items():
                feats.setdefault(k, []).append(v)
    seq_cal = [tuple(card["calibration"]["seq"][k]) for k in LABELS]
    p_any = calibrate(val_pred["seq"]["logits"], seq_cal)[:, 0]
    card["drift_reference"] = build_reference(
        {k: np.asarray(v) for k, v in feats.items()} | {"p_any": p_any}
    )
    card["provenance"] = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "train_config": cfg,
        "best_epoch": ck["epoch"],
        "onnx_parity_max_abs": parity,
        "quantization": {
            "format": "QDQ",
            "weights": "int8 per-channel",
            "activations": "uint8, percentile 99.99",
            "fp32_nodes": excluded,
            "calibration_slices": args.calib_slices,
        },
        "data": json.loads((data_dir / "manifest.json").read_text()) | {"regions": "omitted"},
    }
    card["intended_use"] = (
        "Research and engineering demonstration only. Not a medical device; not validated for "
        "clinical use. Trained on a 2,065-series sample of RSNA 2019 (single source)."
    )
    write_card(stage, card)
    shutil.rmtree(args.out, ignore_errors=True)
    shutil.copytree(stage, args.out)
    (args.out.parent / "export_report.json").write_text(
        json.dumps({p: {h: results[p][h]["test"] for h in results[p]} for p in results}, indent=2)
    )
    for p in results:
        for h in ("cnn", "seq"):
            t = results[p][h]["test"]
            print(
                f"{p}/{h}: macro AUC {t['auc_macro']:.4f}  wlogloss {t['weighted_log_loss']:.4f}"
                f"  study AUC {t['study_any']['auc']:.4f}"
            )
    sizes = {f: (work / f).stat().st_size / 1e6 for f in ("cnn.fp32.onnx", "cnn.int8.onnx")}
    print("sizes MB:", {k: round(v, 1) for k, v in sizes.items()})
    print(f"bundle written to {args.out}")


if __name__ == "__main__":
    main()
