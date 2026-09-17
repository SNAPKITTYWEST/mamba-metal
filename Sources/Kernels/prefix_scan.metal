// prefix_scan.metal
// Inter-tile carry propagation for sequences longer than one threadgroup.
// After each tile has produced a local inclusive scan starting from identity,
// this kernel (or a host reduction) stitches the tiles together.

#include <metal_stdlib>
using namespace metal;

struct Affine {
    float a, b;
};

inline Affine compose(Affine t2, Affine t1) {
    return {t2.a * t1.a, t2.a * t1.b + t2.b};
}

// tile_carries : [num_tiles, B, D, N, 2]  (a,b) for the last element of each tile
// After this kernel the carries become the exclusive prefix for the next tile.
kernel void propagate_tile_carries_f32(
    device float *tile_carries [[buffer(0)]],  // in/out
    constant uint &num_tiles   [[buffer(1)]],
    constant uint &stride      [[buffer(2)]],  // B*D*N*2
    uint gid [[thread_position_in_grid]])
{
    if (gid >= stride) return;

    Affine acc = {1.0f, 0.0f};
    for (uint t = 0; t < num_tiles; ++t) {
        uint idx = t * stride + gid;
        float a = tile_carries[idx];
        float b = tile_carries[idx + 1];       // assuming interleaved or adjust
        // For simplicity we treat the buffer as pairs; real layout may differ
        Affine cur = {a, b};
        Affine new_acc = compose(cur, acc);
        // store exclusive carry for this tile
        tile_carries[idx]     = acc.a;
        tile_carries[idx + 1] = acc.b;
        acc = new_acc;
    }
}
