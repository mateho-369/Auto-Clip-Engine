"""One-command RVC training wrapper for the RVC-Project mainline
(the fork README-STUDIO.md has users install; see
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).

Why this exists: that fork's own "one-click training" only runs from inside
its Gradio UI — there's no bundled single-command CLI equivalent, and its
train/*.py scripts break with a circular-import error when invoked as bare
scripts (running `python train/preprocess.py` puts `train/` itself on
sys.path, which collides with the `train` *package* that same script imports
from). Both are worked around here: every step below runs via `-m` module
invocation instead of a script path, and this file chains the exact sequence
RVC-WebUI's own "train1key" button runs internally (preprocess -> extract F0
-> extract HuBERT features -> write filelist+config -> train -> train index)
so ai_studio's Settings -> Voice timbre -> "train_command" can point at ONE
command instead of five.

Usage:
    python rvc_autotrain.py --rvc-dir <path> --exp <name> --dataset <wav-or-dir>
"""
import argparse
import json
import os
import subprocess
import sys
from random import shuffle


def run(py, rvc_dir, module, args, env):
    cmd = [py, "-m", module] + [str(a) for a in args]
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=rvc_dir, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, flush=True)
        raise SystemExit(f"{module} failed with exit code {proc.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvc-dir", required=True, help="RVC-WebUI install directory")
    ap.add_argument("--exp", required=True, help="experiment / voice profile name")
    ap.add_argument("--dataset", required=True,
                    help="a .wav training sample, or a folder of them")
    ap.add_argument("--sr", default="40k", choices=["32k", "40k", "48k"])
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--f0-method", default="rmvpe", choices=["pm", "rmvpe"])
    ap.add_argument("--total-epoch", type=int, default=200)
    ap.add_argument("--save-every-epoch", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    rvc_dir = os.path.abspath(args.rvc_dir)
    py = os.path.join(rvc_dir, ".venv", "Scripts", "python.exe")
    if not os.path.isfile(py):
        py = os.path.join(rvc_dir, ".venv", "bin", "python")
    if not os.path.isfile(py):
        py = sys.executable  # last resort: hope the caller's own interpreter has RVC's deps

    dataset_dir = args.dataset if os.path.isdir(args.dataset) else os.path.dirname(args.dataset)
    exp_dir = os.path.join(rvc_dir, "logs", args.exp)
    os.makedirs(exp_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = rvc_dir
    env["PYTHONIOENCODING"] = "utf-8"

    sr_hz = {"32k": 32000, "40k": 40000, "48k": 48000}[args.sr]

    run(py, rvc_dir, "train.preprocess",
        [dataset_dir, sr_hz, os.cpu_count() or 4, f"logs/{args.exp}", "False", 3.0], env)

    run(py, rvc_dir, "train.dataset.extract_f0",
        ["cuda", 1, 0, 0, exp_dir, "True"], env)

    run(py, rvc_dir, "train.dataset.extract_hubert_feature",
        ["cuda:0", 1, 0, 0, exp_dir, args.version, "True"], env)

    # --- filelist + config.json (mirrors webui.py's click_train, single-speaker only) ---
    fea_dim = 768 if args.version == "v2" else 256
    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    feature_dir = os.path.join(exp_dir, f"3_feature{fea_dim}")
    f0_dir = os.path.join(exp_dir, "2a_f0")
    f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
    names = (set(n.split(".")[0] for n in os.listdir(gt_wavs_dir))
             & set(n.split(".")[0] for n in os.listdir(feature_dir))
             & set(n.split(".")[0] for n in os.listdir(f0_dir))
             & set(n.split(".")[0] for n in os.listdir(f0nsf_dir)))
    if not names:
        raise SystemExit("no audio survived preprocessing + feature extraction — "
                         "check the sample isn't silent/too short")
    opt = []
    for name in sorted(names):
        opt.append("%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|0" % (
            gt_wavs_dir.replace("\\", "\\\\"), name, feature_dir.replace("\\", "\\\\"), name,
            f0_dir.replace("\\", "\\\\"), name, f0nsf_dir.replace("\\", "\\\\"), name))
    for _ in range(2):
        opt.append("%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|"
                   "%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|0"
                   % (rvc_dir, args.sr, rvc_dir, fea_dim, rvc_dir, rvc_dir))
    shuffle(opt)
    with open(os.path.join(exp_dir, "filelist.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(opt))

    config_path = os.path.join(rvc_dir, "configs", "v1", f"{args.sr}.json")  # 40k/32k/48k v1
    # configs/v2 only ships 32k.json + 48k.json — 40k uses v1's config regardless of
    # model version (this matches webui.py's own "sr2 == '40k'" special case).
    if args.version == "v2" and args.sr != "40k":
        alt = os.path.join(rvc_dir, "configs", "v2", f"{args.sr}.json")
        if os.path.isfile(alt):
            config_path = alt
    with open(config_path, encoding="utf8") as f:
        config_data = json.load(f)
    config_data.pop("speaker_info", None)
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n")

    run(py, rvc_dir, "train.train",
        ["-e", args.exp, "-sr", args.sr, "-f0", "1", "-bs", str(args.batch_size), "-g", "0",
         "-te", str(args.total_epoch), "-se", str(args.save_every_epoch),
         "-pg", f"assets/pretrained_v2/f0G{args.sr}.pth",
         "-pd", f"assets/pretrained_v2/f0D{args.sr}.pth",
         "-l", "0", "-c", "0", "-sw", "1", "-v", args.version], env)

    run(py, rvc_dir, "train.train_index",
        [args.exp, args.version, "assets/indices", os.cpu_count() or 4, "single"], env)

    print(f"\nDone. Weights: {rvc_dir}/assets/weights/{args.exp}.pth", flush=True)


if __name__ == "__main__":
    main()
