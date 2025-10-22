def predict_and_accumulate_metrics(model, filepath, cols, chunksize, start_row, batch_size):
    """Stream prediction; accumulate coverage & mean violation (no huge arrays kept)."""
    t_offset = start_row
    total_points = 0
    total_inside = 0
    violation_sum = 0.0

    for df in read_chunks(filepath, cols, chunksize, start_row=start_row, max_rows=None):
        x = df.values.astype(np.float32)  # (n, F)
        t = np.arange(t_offset, t_offset + len(df), dtype=np.float32).reshape(-1, 1)
        t_mat = np.repeat(t, x.shape[1], axis=1).astype(np.float32)

        # predict in sub-batches
        ds = tf.data.Dataset.from_tensor_slices((x, t_mat)).batch(512)
        o_hi_list, o_lo_list = [], []
        for xb, tb in ds:
            out = model.predict(xb.numpy(), tb.numpy(), batch_size=512)
            if model.synchronize:
                _, _, hi_i, lo_i = out    # 4-tuple when synchronize=True
            else:
                hi_i, lo_i = out          # 2-tuple when synchronize=False

            # Reduce across estimators if present: (batch, n_estimators, F) -> (batch, F)
            if hi_i.ndim == 3:
                hi_i = np.median(hi_i, axis=1)
                lo_i = np.median(lo_i, axis=1)

            o_hi_list.append(hi_i)
            o_lo_list.append(lo_i)

        o_hi = np.concatenate(o_hi_list, axis=0)  # (n, F)
        o_lo = np.concatenate(o_lo_list, axis=0)  # (n, F)  

        pos_hi = np.maximum(0.0, x - o_hi)
        pos_lo = np.maximum(0.0, o_lo - x)
        excess = pos_hi + pos_lo                          # (n, F)

        inside = (excess == 0.0).sum()
        viol = excess[excess > 0.0].sum() if np.any(excess > 0.0) else 0.0

        total_inside  += int(inside)
        total_points  += excess.size
        violation_sum += float(viol)
        t_offset += len(df)

    coverage = total_inside / total_points if total_points > 0 else 0.0
    denom = (total_points - total_inside)
    mean_violation = violation_sum / denom if denom > 0 else 0.0
    return coverage, mean_violation


def evaluate_pointwise_from_predictions(pred_csv_path, segments, chunksize=1_000_000):
    """
    Streaming point-wise precision/recall/F1 using the predictions CSV we wrote:
    columns: idx, score, is_anom
    """
    import numpy as np, pandas as pd

    # compact list of segments; iterate once
    segs = segments
    seg_ptr = 0
    n_segs = len(segs)

    TP = FP = FN = 0

    for df in pd.read_csv(pred_csv_path, chunksize=chunksize):
        idx = df["idx"].to_numpy(dtype=np.int64)
        pred = (df["is_anom"].to_numpy(dtype=np.int8) > 0)
        if len(idx) == 0:
            continue

        lo, hi = int(idx[0]), int(idx[-1] + 1)
        truth = np.zeros(hi - lo, dtype=bool)

        # mark truth by intersecting segments that overlap [lo, hi)
        while seg_ptr > 0 and segs[seg_ptr-1][1] > lo:
            seg_ptr -= 1
        p = seg_ptr
        while p < n_segs and segs[p][0] < hi:
            s, e = segs[p]
            a, b = max(s, lo), min(e, hi)
            if a < b:
                truth[a - lo : b - lo] = True
            p += 1

        # align pred to [lo,hi)
        # our idx is contiguous from lo..hi-1 in the CSV chunks
        TP += int(np.sum(pred & truth))
        FP += int(np.sum(pred & ~truth))
        FN += int(np.sum((~pred) & truth))

        # advance pointer to first segment with end > hi
        while seg_ptr < n_segs and segs[seg_ptr][1] <= hi:
            seg_ptr += 1

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1
