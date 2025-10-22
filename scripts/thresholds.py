def choose_threshold_by_contamination(scores_iter, expected_contam=0.038, sample_size=500_000, seed=42):
    """
    Pick a threshold tau so that about `expected_contam` fraction are positive.
    Uses a memory-safe reservoir sample (approximate quantile).
    """
    import numpy as np, random
    rng = random.Random(seed)

    # reservoir
    R = None
    n_seen = 0
    for _, _, s in scores_iter:
        s = np.asarray(s)
        if R is None:
            R = s[:min(len(s), sample_size)].copy()
            n_seen = len(s)
        else:
            for v in s:
                n_seen += 1
                if len(R) < sample_size:
                    R = np.append(R, v)
                else:
                    j = rng.randrange(n_seen)
                    if j < sample_size:
                        R[j] = v

    if R is None or len(R) == 0:
        return 0.0

    q = np.quantile(R, 1.0 - expected_contam)  # approximate global (1-contam) quantile
    return float(q)

def choose_tau_on_nominal_window(model, filepath, cols, start_row, rows,
                                 chunksize=200_000, batch_size=512, agg="linf",
                                 q=0.995):
    """
    Pick tau as the q-quantile of scores computed on a *nominal-only* window.
    """
    import numpy as np
    from random import Random

    rng = Random(42)
    sample_size = 500_000
    R = None
    n_seen = 0

    for lo, hi, scores in stream_violation_scores(
        model, filepath, cols,
        start_row=start_row, chunksize=chunksize, batch_size=batch_size, agg=agg
    ):
        # only use [start_row, start_row+rows)
        if hi <= start_row + rows:
            s = np.asarray(scores)
            if R is None:
                R = s[:min(len(s), sample_size)].copy()
                n_seen = len(s)
            else:
                for v in s:
                    n_seen += 1
                    if len(R) < sample_size:
                        R = np.append(R, v)
                    else:
                        j = rng.randrange(n_seen)
                        if j < sample_size:
                            R[j] = v
        else:
            break

    if R is None or len(R) == 0:
        return 0.0

    return float(np.quantile(R, q))

def choose_tau_by_max_f1_on_dev(model, filepath, cols, segments,
                                dev_start, dev_rows,
                                chunksize=200_000, batch_size=512, agg="linf",
                                sample_size=600_000, seed=42):
    """
    Stream scores & labels on a labeled dev window and pick tau that maximizes F1.
    Uses a reservoir sample to keep memory bounded.
    """
    import numpy as np, random
    rng = random.Random(seed)

    # 1) Reservoir sample of (score, label) pairs from the dev window
    S = None    # scores
    L = None    # labels (bool)
    n_seen = 0

    # Convert segment list into a simple iterator
    segs = segments
    p = 0
    nsegs = len(segs)

    dev_end = dev_start + dev_rows
    for lo, hi, scores in stream_violation_scores(model, filepath, cols,
                                                  start_row=dev_start, chunksize=chunksize,
                                                  batch_size=batch_size, agg=agg):
        if lo >= dev_end:
            break
        hi = min(hi, dev_end)

        idx = np.arange(lo, hi, dtype=np.int64)
        s   = np.asarray(scores, dtype=np.float32)

        # Build truth mask for [lo, hi)
        truth = np.zeros(len(idx), dtype=bool)
        # Advance p to first seg that might overlap
        while p < nsegs and segs[p][1] <= lo:
            p += 1
        q = p
        while q < nsegs and segs[q][0] < hi:
            a = max(segs[q][0], lo) - lo
            b = min(segs[q][1], hi) - lo
            if a < b:
                truth[a:b] = True
            q += 1

        # Reservoir sample
        if S is None:
            take = min(len(s), sample_size)
            S = s[:take].copy()
            L = truth[:take].copy()
            n_seen = len(s)
        else:
            for si, li in zip(s, truth):
                n_seen += 1
                if len(S) < sample_size:
                    S = np.append(S, si)
                    L = np.append(L, li)
                else:
                    j = rng.randrange(n_seen)
                    if j < sample_size:
                        S[j] = si
                        L[j] = li

    if S is None or len(S) == 0:
        return 0.0

    # 2) Scan a grid of percentiles to maximize F1
    percentiles = np.linspace(80, 99.9, 60)  # coarse→fine
    best_tau, best_f1 = None, -1.0
    for pctl in percentiles:
        tau = float(np.percentile(S, pctl))
        pred = (S >= tau)
        TP = int(np.sum(pred & L))
        FP = int(np.sum(pred & ~L))
        FN = int(np.sum((~pred) & L))
        prec = TP / (TP + FP) if (TP+FP) else 0.0
        rec  = TP / (TP + FN) if (TP+FN) else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    print(f"[dev-thresh] best F1={best_f1:.4f} at tau={best_tau:.6f}")
    return best_tau
