"""CPRP insights analysis — turns Make 'CPRP Insights Snapshots' records into comparable KPIs.

Input: JSON file = list of records as returned by Make data-store-records_list for data store
240581 (each {key, data:{media_id, posted_at, captured_at, permalink, caption_head, metrics}}).
Optional --register JSON: {"<media_id>": {"producer": "claude", "episode": "...", "hook": "...", "theme": "..."}}.
Optional --duration seconds (default 15).

Prints a KPI table (latest snapshot per reel), age-matched comparison and producer split.
Analytics output is PRIVATE: never commit results to the public repo.
"""
import argparse, json, re, statistics as st
from datetime import datetime

def parse_metrics(s):
    out = {}
    for k, v in re.findall(r"([a-z_]+)=([-0-9.]*);", s or ""):
        try: out[k] = float(v) if v != "" else None
        except ValueError: out[k] = None
    return out

def iso(s):
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)

def kpis(rec, duration):
    m = parse_metrics(rec["metrics"])
    reach = m.get("reach") or 0
    views = m.get("views") or 0
    age_h = (iso(rec["captured_at"]) - iso(rec["posted_at"])).total_seconds() / 3600
    avg_w = (m.get("ig_reels_avg_watch_time") or 0) / 1000
    shares, saves, reposts = m.get("shares") or 0, m.get("saved") or 0, m.get("reposts") or 0
    per_k = (lambda x: 1000 * x / reach if reach else None)
    return {
        "media_id": rec["media_id"], "age_h": age_h, "views": views, "reach": reach,
        "hook_hold_%": None if m.get("reels_skip_rate") is None else 100 - m["reels_skip_rate"],
        "skip_%": m.get("reels_skip_rate"),
        "avg_watch_s": avg_w, "watch_ratio": avg_w / duration if duration else None,
        "shares_per_1k": per_k(shares), "saves_per_1k": per_k(saves),
        "send_save_repost_per_1k": per_k(shares + saves + reposts),
        "likes": m.get("likes") or 0, "comments": m.get("comments") or 0,
        "views_per_reach": views / reach if reach else None,
        "caption_head": rec["caption_head"].split("\n")[0][:60], "posted_at": rec["posted_at"],
    }

def fmt(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records"); ap.add_argument("--register"); ap.add_argument("--duration", type=float, default=15)
    ap.add_argument("--min-age", type=float, default=20, help="min age (h) for comparisons")
    a = ap.parse_args()
    recs = json.load(open(a.records))
    recs = [r["data"] if "data" in r else r for r in recs]
    reg = json.load(open(a.register)) if a.register else {}
    # latest snapshot per media + snapshot closest to 24 h
    by = {}
    for r in recs: by.setdefault(r["media_id"], []).append(r)
    rows, rows24 = [], []
    for mid, rs in by.items():
        rs.sort(key=lambda r: r["captured_at"])
        rows.append(kpis(rs[-1], a.duration))
        k = [kpis(r, a.duration) for r in rs]
        best = min(k, key=lambda x: abs(x["age_h"] - 24))
        if 12 <= best["age_h"] <= 48: rows24.append(best)
    rows.sort(key=lambda r: r["posted_at"], reverse=True)
    cols = ["age_h", "views", "reach", "hook_hold_%", "avg_watch_s", "watch_ratio", "send_save_repost_per_1k", "likes", "producer", "caption_head"]
    print("LATEST SNAPSHOT PER REEL (newest first)")
    print(" | ".join(cols))
    for r in rows:
        r["producer"] = reg.get(r["media_id"], {}).get("producer", "other")
        print(" | ".join(fmt(r.get(c), 2 if c == "watch_ratio" else 1) for c in cols))
    mature = [r for r in rows if r["age_h"] >= a.min_age and r["reach"]]
    def summary(rs, label):
        if not rs: print(f"{label}: n=0"); return
        g = lambda k: [r[k] for r in rs if r[k] is not None]
        print(f"{label}: n={len(rs)} | median views {st.median(g('views')):.0f} | median hook hold {st.median(g('hook_hold_%')):.1f}% "
              f"| median avg watch {st.median(g('avg_watch_s')):.1f}s | mean send+save+repost/1k {st.mean(g('send_save_repost_per_1k')):.1f}")
    print(); print(f"MATURE REELS (age ≥ {a.min_age:.0f} h)")
    summary(mature, "all")
    for p in sorted({r['producer'] for r in mature}):
        summary([r for r in mature if r["producer"] == p], f"producer={p}")
    if rows24:
        print(); print("AGE-MATCHED (~24 h snapshot, 12–48 h window)")
        summary(rows24, "all@24h")
    if len(mature) >= 4:
        top = max(mature, key=lambda r: (r["hook_hold_%"] or 0) * (r["avg_watch_s"] or 0))
        low = min(mature, key=lambda r: (r["hook_hold_%"] or 0) * (r["avg_watch_s"] or 0))
        print(); print(f"STRONGEST retention: {top['caption_head']} (hold {fmt(top['hook_hold_%'])}%, watch {fmt(top['avg_watch_s'])}s)")
        print(f"WEAKEST retention:   {low['caption_head']} (hold {fmt(low['hook_hold_%'])}%, watch {fmt(low['avg_watch_s'])}s)")
    print("\nCaveats: small n, unequal ages/posting times, no causal claims; watch_ratio = avg watch / length (not completion rate).")

if __name__ == "__main__":
    main()
