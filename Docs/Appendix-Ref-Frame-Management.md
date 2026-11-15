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
| `sframe_dist` / `sframe_posi` | unset | An S-frame must refresh all eight DPB slots, which evicts every anchor on the decoder side only. Rejected in combination -- see section 3's S-frame note. |
| preset | one whose reference counts are all <= 2 | The anchor pool is the top 4 DPB slots, so the encoder's own references must fit in the bottom 4. Rejected otherwise -- see section 5. |
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

S-frame interaction: the two features are mutually exclusive and the
combination is rejected at configuration time. An S-frame refreshes all
eight DPB slots and carries no `refresh_frame_flags`, so a decoder always
evicts every anchor, while the encoder's Phase-3 guard keeps the anchor
slots out of the refresh. The two DPB views would diverge from the
S-frame onwards, and a later USE would predict from a different picture
at each end.

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

## 5. Encoder-side mechanics (DPB slot split)

The DPB is partitioned rather than negotiated. The encoder's own
references own slots {0..3}; application anchors own {4..7}. The pool is
a fixed `0xF0`, returned by `svt_aom_ref_mgmt_storeable_slots_mask` in
`pd_process.c`, and STORE never selects outside it.

That split is only representable while the encoder's references stay in
the bottom 4, which holds exactly when every reference-list count is
<= 2 -- i.e. `mrp_ctrls.ld_reduce_ref_buffs >= 1`. At `ld_reduce == 0`
the encoder uses 3 references, `prune_refs` no longer collapses `LAST3`
onto `LAST`, and `LAST3 = long_base_idx = 7` is a live reference inside
the anchor pool.

`ld_reduce_ref_buffs` is derived from the preset's `mrp_level` in
`set_mrp_ctrl` and is fixed for the session. Rather than override the
preset to make LTR fit, `svt_av1_enc_set_parameter` rejects the
combination:

```
if (max_managed_refs > 0 && mrp_ctrls.ld_reduce_ref_buffs == 0)
    return EB_ErrorBadParameter;
```

Which low-delay CBR presets qualify:

| Config | `mrp_level` | list counts | `ld_reduce` | LTR |
|---|---|---|---|---|
| `rtc`, M9+ | 0 | 1/0/1/0 | 2 | accepted |
| non-`rtc`, M10+ | 0 or 11 | 1/0/1/0 | 2 | accepted |
| non-`rtc`, M9 | 9 | 3/2/1/1 | 0 | rejected |
| `rtc`, M7/M8 | 6 | 3/2/3/2 | 0 | rejected |
| non-`rtc`, <= M8 | 1/2/4 | 4/3/4/3 | 0 | rejected |

So in practice an LTR session runs single-reference, because that is
what the qualifying presets select -- not because LTR constrains it.
The anchors occupy slots the RPS was already using only as its per-frame
`| 0xf0` / `| 0xfc` scrub-and-backup bits, never as prediction
references, so the split costs the encoder no usable reference capacity.

Mid-stream `PRESET_CHANGE_EVENT` does NOT re-run `set_mrp_ctrl`. The DPB
layout (`ld_reduce_ref_buffs`, `flat_max_refs`, lay0/lay1 toggle ranges,
buffer allocations) stays locked at init, which is what keeps the gate's
verdict valid for the life of the session. The per-frame ref counts that
mode decision consumes ARE updated, via
`svt_aom_clamp_mrp_ctrls_to_runtime_preset`, clamped against the init
snapshot `scs->mrp_ctrls_init`:

```
mrp_ctrls.base_ref_list0_count =
    MIN(mrp_ctrls_init.base_ref_list0_count,
        runtime_preset_natural_base_list0);
```

**Caller contract: initialize at the slowest preset you will ever reach
mid-stream.** The init snapshot is the upper envelope; runtime presets
shrink within it but cannot grow past it, because pools and toggle
ranges are sized at init.

Resize-driven reinits (`svt_av1_enc_deinit_handle` +
`svt_av1_enc_init_handle` + `svt_av1_enc_init`) re-run the whole init
path, including a fresh snapshot, so a deinit/init cycle resets the
envelope. Only `PRESET_CHANGE_EVENT` in isolation is bounded by it.

## 6. Memory overhead

When `max_managed_refs > 0`, the encoder's reference-picture pool grows
by `max_managed_refs` extra buffers (see `min_ref += max_managed_refs`
in `enc_handle.c`). Same for the PA-ref pool. With
`max_managed_refs = 0` (default) the legacy memory footprint is
preserved bit-exactly.
