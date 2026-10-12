import numpy as np


def im2col(x, kernel_h, kernel_w, stride=1, padding=0):
    """
    Convert NCHW input tensor into a 2D matrix for GEMM-based convolution.

    Input:
        x shape: [N, C, H, W]

    Output:
        cols shape: [C * kernel_h * kernel_w, N * out_h * out_w]

    Each column is one flattened convolution window.
    """
    n, c, h, w = x.shape
    out_h = (h + 2 * padding - kernel_h) // stride + 1
    out_w = (w + 2 * padding - kernel_w) // stride + 1

    # Zero padding on height and width dimensions.
    x_padded = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )

    cols = np.zeros((c * kernel_h * kernel_w, n * out_h * out_w), dtype=x.dtype)

    col_idx = 0
    for batch in range(n):
        for oh in range(out_h):
            for ow in range(out_w):
                h_start = oh * stride
                w_start = ow * stride
                patch = x_padded[
                    batch,
                    :,
                    h_start : h_start + kernel_h,
                    w_start : w_start + kernel_w,
                ]
                cols[:, col_idx] = patch.reshape(-1)
                col_idx += 1

    return cols, out_h, out_w


def conv2d_im2col_gemm(x, weight, stride=1, padding=0):
    """
    Run convolution through im2col + GEMM.

    weight shape: [M, C, R, S]
    output shape: [N, M, out_h, out_w]
    """
    n = x.shape[0]
    m, c, r, s = weight.shape

    x_col, out_h, out_w = im2col(x, r, s, stride=stride, padding=padding)

    # Flatten filters into GEMM matrix:
    #   W_col: [M, C * R * S]
    #   X_col: [C * R * S, N * out_h * out_w]
    #   Y_col: [M, N * out_h * out_w]
    w_col = weight.reshape(m, c * r * s)
    y_col = w_col @ x_col

    # Convert GEMM output back to NCHW layout.
    y = y_col.reshape(m, n, out_h, out_w).transpose(1, 0, 2, 3)
    return y, x_col


def cross_filter_structured_pruning(weight, tm=4, prune_ratio=0.5):
    """
    Cross-filter structured pruning.

    For every cluster of Tm output filters:
      1. Compute one shared importance map over [C, R, S].
      2. Prune the same low-importance positions for every filter in the cluster.

    importance[c, r, s] = sum(abs(W[m, c, r, s])) over filters in the cluster
    """
    m, c, r, s = weight.shape
    positions_per_filter = c * r * s
    num_prune = int(np.floor(prune_ratio * positions_per_filter + 0.5))
    num_prune = min(max(num_prune, 0), positions_per_filter)

    masks = []
    w_pruned = weight.copy()

    for cluster_start in range(0, m, tm):
        cluster_end = min(cluster_start + tm, m)
        cluster = weight[cluster_start:cluster_end]

        # Shared [C, R, S] importance map for this filter cluster.
        importance = np.sum(np.abs(cluster), axis=0)
        flat_importance = importance.reshape(-1)

        # Pick positions with the smallest importance values.
        prune_indices = np.argsort(flat_importance)[:num_prune]

        flat_mask = np.ones(positions_per_filter, dtype=np.float32)
        flat_mask[prune_indices] = 0.0
        mask = flat_mask.reshape(c, r, s)
        masks.append(mask)

        # Broadcast the same [C, R, S] mask to all filters in this cluster.
        w_pruned[cluster_start:cluster_end] *= mask

    return w_pruned, masks


def count_effective_macs(masks, output_spatial_size, tm=4, total_filters=8):
    """
    Count MACs after structured pruning.

    Each kept [C, R, S] position is used by every filter in the cluster and
    every output spatial position.
    """
    effective_macs = 0
    for cluster_id, mask in enumerate(masks):
        cluster_start = cluster_id * tm
        cluster_end = min(cluster_start + tm, total_filters)
        filters_in_cluster = cluster_end - cluster_start
        kept_positions = int(np.sum(mask))
        effective_macs += kept_positions * filters_in_cluster * output_spatial_size
    return effective_macs


def main():
    # Reproducible random data.
    np.random.seed(0)

    # Problem setting.
    n, c, h, w = 1, 3, 8, 8
    m, r, s = 8, 3, 3
    stride = 1
    padding = 1
    tm = 4
    prune_ratio = 0.5

    # 1. Create random input and weight.
    x = np.random.randn(n, c, h, w).astype(np.float32)
    weight = np.random.randn(m, c, r, s).astype(np.float32)

    # 2-3. Original im2col + GEMM convolution.
    y_original, x_col = conv2d_im2col_gemm(x, weight, stride=stride, padding=padding)

    # 4-5. Cross-filter structured pruning and masked weight.
    weight_pruned, masks = cross_filter_structured_pruning(
        weight,
        tm=tm,
        prune_ratio=prune_ratio,
    )

    # 6. Pruned im2col + GEMM convolution.
    y_pruned, _ = conv2d_im2col_gemm(x, weight_pruned, stride=stride, padding=padding)

    # 7. Statistics and output error.
    original_weight_count = weight.size
    pruned_nonzero_weight_count = int(np.count_nonzero(weight_pruned))
    actual_pruning_ratio = 1.0 - pruned_nonzero_weight_count / original_weight_count

    out_h, out_w = y_original.shape[2], y_original.shape[3]
    output_spatial_size = n * out_h * out_w
    original_macs = n * out_h * out_w * m * c * r * s
    pruned_effective_macs = count_effective_macs(
        masks,
        output_spatial_size=output_spatial_size,
        tm=tm,
        total_filters=m,
    )
    mac_reduction = 1.0 - pruned_effective_macs / original_macs

    abs_error = np.abs(y_original - y_pruned)
    mean_abs_error = float(np.mean(abs_error))
    max_abs_error = float(np.max(abs_error))

    print("=== Cross-filter structured pruning + im2col + GEMM simulation ===")
    print(f"Input X shape: {x.shape}")
    print(f"Weight W shape: {weight.shape}")
    print(f"im2col X_col shape: {x_col.shape}")
    print(f"Output Y shape: {y_original.shape}")
    print(f"Expected output shape: {(1, 8, 8, 8)}")
    print()

    print("=== Cluster masks, each shape = [C, R, S] ===")
    for cluster_id, mask in enumerate(masks):
        cluster_start = cluster_id * tm
        cluster_end = min(cluster_start + tm, m)
        kept = int(np.sum(mask))
        pruned = mask.size - kept
        print(f"Cluster {cluster_id}: filters [{cluster_start}, {cluster_end})")
        print(f"kept positions: {kept}, pruned positions: {pruned}")
        print(mask.astype(np.int32))
        print()

    print("=== Comparison ===")
    print(f"Original weight count: {original_weight_count}")
    print(f"Pruned nonzero weight count: {pruned_nonzero_weight_count}")
    print(f"Actual pruning ratio: {actual_pruning_ratio:.2%}")
    print(f"Original MACs: {original_macs}")
    print(f"Pruned effective MACs: {pruned_effective_macs}")
    print(f"MAC reduction: {mac_reduction:.2%}")
    print(f"Output mean absolute error: {mean_abs_error:.6f}")
    print(f"Output max absolute error: {max_abs_error:.6f}")


if __name__ == "__main__":
    main()
