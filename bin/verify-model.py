#!/usr/bin/env python3
"""Verify a downloaded HF model's on-disk integrity.

In the HF cache, every large (LFS) file is stored as blobs/<sha256-of-content> with a snapshot
symlink pointing to it. So a file is intact iff sha256(content) == its blob filename. This catches
the failure mode that bit us twice: an interrupted download leaving a full-size blob with unwritten
(zero) regions — right size, wrong content — which hf's presence/size checks miss and only explodes
at load time.

  verify-model.py <hf-repo-id> [--cache DIR]                 # exit 0 if clean, 1 if any corrupt
  verify-model.py <hf-repo-id> [--cache DIR] --delete-bad    # also rm corrupt blobs (hf refetches)

Small non-LFS files (config/tokenizer) use git-sha1 blob names (40 hex) and are skipped — only the
weight shards (sha256, 64 hex) are content-verified.
"""
import sys, os, hashlib, glob, argparse


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--cache", required=True, help="model cache root (cluster.model_cache)")
    ap.add_argument("--delete-bad", action="store_true",
                    help="delete corrupt/mismatched blobs so a re-download re-fetches them")
    a = ap.parse_args()

    d = os.path.join(a.cache, "hub", "models--" + a.model.replace("/", "--"))
    snaps = sorted(glob.glob(os.path.join(d, "snapshots", "*")))
    if not snaps:
        print(f"NO SNAPSHOT for {a.model} under {d}")
        sys.exit(2)
    snap = snaps[-1]

    bad, checked = [], 0
    for name in sorted(os.listdir(snap)):
        real = os.path.realpath(os.path.join(snap, name))
        blob = os.path.basename(real)
        # only LFS files (sha256 = 64 hex) carry content-addressed names we can verify
        if len(blob) != 64 or any(c not in "0123456789abcdef" for c in blob):
            continue
        if not os.path.exists(real):
            bad.append((name, real, "MISSING"))
            continue
        checked += 1
        if sha256_file(real) != blob:
            bad.append((name, real, "SHA256_MISMATCH"))

    if bad:
        for name, real, why in bad:
            print(f"BAD {name} ({why})")
            if a.delete_bad and why == "SHA256_MISMATCH" and os.path.exists(real):
                os.remove(real)
                print(f"    deleted {os.path.basename(real)}")
        print(f"FAIL: {len(bad)} corrupt/missing of {checked + len(bad)} weight shards for {a.model}")
        sys.exit(1)
    print(f"OK: {checked} weight shards verified for {a.model}")


if __name__ == "__main__":
    main()
