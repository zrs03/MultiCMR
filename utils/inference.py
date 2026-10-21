"""Reconstruct native LAX frames and every SAX (slice, phase) center."""
import torch
from preprocess.processing import paired_centers, window_indices, sax_context_indices, validate_sax_geometry

VIEWS = ('2ch', '4ch', 'sa')


@torch.inference_mode()
def predict_cine(model, images, modality_mask, fp16=False, sax_phases=None):
    """Full inputs [B,1,F,H,W]; SAX uses canonical flat index z*T+t.

    The phase count is required metadata, never inferred from image contents.
    Average duplicate predictions ONLY for the same output frame, never across
    phases. LAX predictions may be averaged over different SAX slice contexts.
    """
    device_type = images['sa'].device.type
    if fp16 and device_type != 'cuda':
        raise ValueError('FP16 inference requires a CUDA device')
    if model.training:
        raise ValueError('predict_cine requires model.eval()')
    if modality_mask.shape != (images['sa'].shape[0], 3) or not modality_mask.bool().any(1).all():
        raise ValueError('Expected nonempty modality_mask [B,3]')
    if any(x.ndim != 5 or x.shape[1] != 1 for x in images.values()):
        raise ValueError('Expected per-view [B,1,F,H,W] inputs')
    slices = validate_sax_geometry(images['sa'].shape[2], sax_phases)
    lengths = {v: images[v].shape[2] for v in VIEWS[:2]}
    lengths['sa'] = sax_phases
    centers = paired_centers(lengths)
    if not modality_mask[:, :2].any():
        centers = [dict(sa=t, **{'2ch': 0, '4ch': 0}) for t in range(sax_phases)]
    elif not modality_mask[:, 2].any():
        centers = [dict(sa=0, **c) for c in paired_centers({v: lengths[v] for v in VIEWS[:2]})]
    sums, counts = {}, {}
    for center in centers:
        lax = {}
        for view in VIEWS[:2]:
            indices = torch.as_tensor(window_indices(center[view], lengths[view], model.temporal_size),
                                      device=images[view].device)
            lax[view] = images[view].index_select(2, indices)
        for z in range(slices if modality_mask[:, 2].any() else 1):
            indices = torch.as_tensor(sax_context_indices(center['sa'], z, sax_phases, slices),
                                      device=images['sa'].device)
            sample = dict(lax, sa=images['sa'].index_select(2, indices))
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=fp16):
                logits = model(sample, modality_mask)['logits']
            for i, view in enumerate(VIEWS):
                if not modality_mask[:, i].any():
                    continue
                probability = logits[view].float().softmax(1).cpu()
                if view not in sums:
                    frames = images[view].shape[2]
                    sums[view] = probability.new_zeros(probability.shape[:2] + (frames,) + probability.shape[-2:])
                    counts[view] = torch.zeros(frames)
                frame = z * sax_phases + center['sa'] if view == 'sa' else center[view]
                sums[view][:, :, frame] += probability
                counts[view][frame] += 1
    predictions = {}
    for i, view in enumerate(VIEWS):
        if view not in sums:
            continue
        if not (counts[view] > 0).all():
            raise RuntimeError(f'Uncovered output frames in {view}')
        pred = (sums[view] / counts[view][None, None, :, None, None]).argmax(1)
        pred[~modality_mask[:, i].bool().cpu()] = 0
        predictions[view] = pred
    return predictions
