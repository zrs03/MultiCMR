"""Reconstruct native-length cines from center-window predictions."""
import torch
from preprocess.processing import paired_centers, window_indices

VIEWS = ('2ch', '4ch', 'sa')


@torch.inference_mode()
def predict_cine(model, images, modality_mask):
    """Slide paired LAX windows with the same SAX; average repeated probabilities.

    Inputs: complete native-length LAX [B,1,T,H,W], SAX [B,1,D,H,W].
    Outputs: LAX [B,T,H,W], SAX [B,D,H,W], CPU. The mask is fixed across
    all windows of a patient. SAX probabilities are averaged before argmax.
    """
    if model.training:
        raise ValueError('predict_cine requires model.eval()')
    if modality_mask.shape != (images['sa'].shape[0], 3):
        raise ValueError('Expected modality_mask [B,3]')
    if any(x.ndim != 5 or x.shape[1] != 1 for x in images.values()):
        raise ValueError('Expected per-view [B,1,T/D,H,W] inputs')
    lengths = {v: images[v].shape[2] for v in VIEWS[:2]}
    centers = paired_centers(lengths)
    if not modality_mask[:, :2].any():
        centers = centers[:1]  # SAX-only predictions do not depend on the window.
    sums, counts = {}, {}
    for center in centers:
        sample = {'sa': images['sa']}
        for view in VIEWS[:2]:
            indices = torch.as_tensor(window_indices(center[view], lengths[view], model.temporal_size),
                                      device=images[view].device)
            sample[view] = images[view].index_select(2, indices)
        logits = model(sample, modality_mask)['logits']
        for i, view in enumerate(VIEWS):
            if not modality_mask[:, i].any():
                continue
            probability = logits[view].float().softmax(1).cpu()
            if view not in sums:
                if view == 'sa':
                    sums[view] = torch.zeros_like(probability)
                    counts[view] = 0
                else:
                    sums[view] = probability.new_zeros(probability.shape[:2] + (lengths[view],) + probability.shape[-2:])
                    counts[view] = torch.zeros(lengths[view])
            if view == 'sa':
                sums[view] += probability
                counts[view] += 1
            else:
                sums[view][:, :, center[view]] += probability
                counts[view][center[view]] += 1
    predictions = {}
    for i, view in enumerate(VIEWS):
        if view not in sums:
            continue
        divisor = counts[view] if view == 'sa' else counts[view][None, None, :, None, None]
        pred = (sums[view] / divisor).argmax(1)
        pred[~modality_mask[:, i].bool().cpu()] = 0
        predictions[view] = pred
    return predictions
