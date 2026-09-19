[Top level](../README.md)

# Application-Controlled Reference Frame Management

## 1. Overview

The ref-frame management API lets an application STORE encoded frames into
the AV1 DPB as long-term references (LTRs), then later instruct the encoder
via USE to predict exclusively from one of those STOREd frames. It is designed for
RTC-style error-recovery: a sender can keep a small pool of known-good
anchor frames, and on packet loss can resynchronize by emitting a delta
that depends only on an acked anchor — much cheaper than a full keyframe.

The encoder normally uses all 8 AV1 DPB slots for short-term references.
When the application STOREs a frame, the encoder takes one of those slots
and locks it — the slot is guarded against being overwritten by the
short-term allocator. The slot is released when the application CLEARs the
corresponding `pic_id`.

The API is exposed via the existing `EbPrivDataNode` side-channel using
three new event types: `REF_STORE_EVENT`, `REF_USE_EVENT`,
`REF_CLEAR_EVENT`. Their payload (`SvtAv1RefFrameCmd`) carries a single
opaque `uint32_t pic_id`. See `EbSvtAv1.h` for the inline declarations.

## 2. Workflow

The typical RTC error-recovery cycle:

1. **Init.** Set `EbSvtAv1EncConfiguration::max_managed_refs` to the
   maximum number of simultaneously-STOREd anchors the application will
   hold (1..4). 0 disables the feature with no memory overhead.

2. **STORE.** Send frame N attached with `REF_STORE_EVENT(pic_id=N)`. The
   encoder places this frame's reconstruction into a STORE-safe DPB slot.

3. **ACK.** Wait for an out-of-band acknowledgement from the far end that
   frame N decoded successfully; record N as a known-good recovery anchor.

4. **RECOVERY (USE).** On notification that some frame after N was lost,
   send the next frame with `REF_USE_EVENT(pic_id=N)`. The encoder
   predicts only from anchor N and refreshes every non-STOREd DPB slot
   with the current frame, giving subsequent frames a clean dependency
   chain.

   USE alone does NOT register a new addressable anchor. To make the
   recovery frame itself CLEARable later, pair the USE with a STORE on
   the same input (using a DIFFERENT `pic_id`).

5. **CLEAR.** Once a newer anchor is acked, send
   `REF_CLEAR_EVENT(pic_id=N)` to release that DPB slot back to the
   short-term allocator.

At most one STORE, one CLEAR, and one USE event is honored per input
`EbBufferHeaderType`. If the same input chains multiple nodes of the same
type, only the FIRST is kept and the rest are dropped with a warning.

## 3. Configuration constraints

| Knob | Required value | Why |
|---|---|---|
| `pred_structure` | `LOW_DELAY` | The ref-mgmt path lives in the LD branch of the prediction-structure generator. |
| `rate_control_mode` | `CBR` | LD-CRF has a different DPB layout (shifted `lay1_offset`) that has not been audited for STORE-pool safety. `enc_settings.c` rejects non-CBR. |
| `hierarchical_levels` | 0 (L1T1) or 1/2 (L1T2/L1T3) | hier >= 3 has not been validated. |
| `max_managed_refs` | 1..4 | ABI cap; matches buffer-pool sizing in `enc_handle.c`. |
| `force_key_frames` | true (recommended) | Required for per-frame `pic_type=KEY` requests in LD; the USE-fallback path depends on it. |

`pic_id` is opaque to the encoder. The application is responsible for
generating unique non-zero values; "unique" means no two currently-STOREd
anchors share a `pic_id`. After CLEAR the id is free to reuse.
`pic_id == 0` is reserved as the "no event" sentinel.

Key frames implicitly release ALL anchors (the encoder refreshes every
DPB slot), so the application should re-issue any STORE it wants to
persist past a KF.

Events apply only on base-layer frames (`temporal_layer_index == 0`).
With `hierarchical_levels = 1` or `2` in LD-CBR mode the application
must track which input frames are base-layer and only attach events to
those — events attached to non-base frames are dropped with a warning.

S-frame interaction: S-frames are NOT treated as automatic anchor resets
by the ref-mgmt layer. If the application emits an S-frame without first
CLEARing its anchors, those anchors remain valid for future USE; whether
that is desirable depends on the application's resync protocol.

## 4. Error handling

There is no output-side confirmation flag; the application's state
machine assumes success. If a precondition is violated, the encoder logs
`SVT_ERROR` and (where possible) fails the call up front.

### 4.1 FAIL-HARD

`svt_av1_enc_send_picture` returns `EB_ErrorBadParameter` and
`EB_BUFFERFLAG_EOS` is stamped on the input buffer (stream is
terminated). Triggered by:

- malformed payload (wrong size or NULL data)
- `pic_id == 0`
- `max_managed_refs == 0` (feature not enabled at init)
- `pred_structure != LOW_DELAY`
- `rate_control_mode != CBR`
- more than one event of the same type on a single input
- `pic_id` collision across STORE / CLEAR / USE on a single input

### 4.2 ERROR + DROP

The encoder logs `SVT_ERROR` and the event has no effect on the
bitstream. These conditions indicate caller misuse that the synchronous
API path cannot detect (they depend on per-frame internal state):

- event on an AV1-overlay frame (not reachable in RTC LD)
- event on a non-base temporal-layer frame
- STORE: `pic_id` already STOREd (must CLEAR first)
- STORE: `max_managed_refs` cap reached (must CLEAR something first)
- STORE: safe slot pool full
- CLEAR / USE: `pic_id` not found in the DPB

The application is expected to maintain its own anchor-state model and
avoid sending events that would hit any ERROR + DROP path.

## 5. Encoder-side mechanics (mrp_level override)

When `max_managed_refs > 0`, the encoder must reserve enough DPB slots
for the app's anchors. The STORE-safe slot pool is a function of
`ld_reduce_ref_buffs` (derived from `mrp_level`'s ref counts in
`set_mrp_ctrl`):

| `ld_reduce_ref_buffs` | Triggered when (LD-CBR) | Safe pool in L1T3 | Effective STOREs |
|---|---|---|---|
| 0 | any list0 count > 2 | {6, 7} | 2 |
| 1 | all list0 counts <= 2 | {0,1,2,4,5,6,7} | 7 (capped at 4) |
| 2 | all list0 counts <= 1 | {0,2,3,4,5,6,7} | 7 (capped at 4) |

The encoder picks its `mrp_level` per-preset before considering LTR. For
the RTC-tuned non-flat-IPP path:

- M9   -> mrp_level 6 (list0 3/3) -> ld_reduce 0 -> 2 safe slots
- M10  -> mrp_level 9 (list0 3/1) -> ld_reduce 0 -> 2 safe slots
- M11+ -> mrp_level 0 (list0 1/1) -> ld_reduce 2 -> 7 safe slots

When the natural pool is insufficient for `max_managed_refs`, the
encoder switches to the cheapest fallback `mrp_level` that satisfies
the constraint, preferring to preserve the `non_base_ref_list0_count`
(non-base refs are consulted by every TID>0 frame — the bulk of an
L1T3 encode):

- M11+ : NO override. Native pool already holds 4 STOREs.
- M10 (level 9) -> level 10 (list0 2/1): base drops 3->2, non_base unchanged.
- M9  (level 6) -> level 8  (list0 2/2): base and non_base each drop by 1.

The override fires once at `svt_av1_enc_init` time and the encoder
snapshots the resulting `mrp_ctrls` to `scs->mrp_ctrls_init`.

Mid-stream `PRESET_CHANGE_EVENT` does NOT re-run `set_mrp_ctrl` — the
DPB layout (`ld_reduce_ref_buffs`, lay0/lay1_toggle ranges, buffer
allocations) stays locked at init values. But the per-frame ref counts
that mode-decision consumes (`mrp_ctrls.base_ref_list0_count` /
`non_base_ref_list0_count`) ARE updated, via
`svt_aom_clamp_mrp_ctrls_to_runtime_preset`:

```
mrp_ctrls.base_ref_list0_count =
    MIN(mrp_ctrls_init.base_ref_list0_count,
        runtime_preset_natural_base_list0);
mrp_ctrls.non_base_ref_list0_count =
    MIN(mrp_ctrls_init.non_base_ref_list0_count,
        runtime_preset_natural_non_base_list0);
```

The clamp is against the INIT snapshot (not the previous runtime
value), so list counts bounce freely up and down within the init
envelope. Example sequence with init at M9 + LTR
(snapshot = list0 2/2):

| step | runtime preset | natural list0 | clamped list0 | matches native? |
|---|---|---|---|---|
| init M9 + LTR | (M9) | 2/2 (mrp_level=8 override) | 2/2 | ✓ |
| -> M10 | M10 | 2/1 (mrp_level=10 natural) | MIN(2/2, 2/1) = 2/1 | ✓ |
| -> M11 | M11 | 1/1 (mrp_level=0 natural) | MIN(2/2, 1/1) = 1/1 | ✓ |
| -> M9 | M9 | 2/2 | MIN(2/2, 2/2) = 2/2 | ✓ — restored |

**Caller contract: initialize at the slowest preset you'll ever reach
mid-stream.** The init snapshot is the upper envelope on ref counts;
runtime presets can shrink within it but cannot grow past it (buffer
pools and DPB toggle ranges are sized at init). If you init at M11
and later switch to M9, the clamp will keep list0 at 1/1 — M9 won't
get its natural 2/2.

Resize-driven encoder reinits (`svt_av1_enc_deinit_handle` +
`svt_av1_enc_init_handle` + `svt_av1_enc_init`) re-run the entire
init path — including a fresh `mrp_ctrls_init` snapshot for the new
init preset — so a deinit/init cycle resets the envelope. Only
`PRESET_CHANGE_EVENT` in isolation is bounded by the snapshot.

The STORE-safe pool itself is derived directly from the LD branch's
exclusive-write slot set — see `svt_aom_ref_mgmt_storeable_slots_mask`
in `pd_process.c`.

## 6. Memory overhead

When `max_managed_refs > 0`, the encoder's reference-picture pool grows
by `max_managed_refs` extra buffers (see `min_ref += max_managed_refs`
in `enc_handle.c`). Same for the PA-ref pool. With
`max_managed_refs = 0` (default) the legacy memory footprint is
preserved bit-exactly.
