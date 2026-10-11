"""Complete-cine inference with framewise 2D encoding and temporal token fusion."""
import torch

VIEWS = ('2ch', '4ch', 'sa')


@torch.inference_mode()
def predict_cine(model, images, modality_mask):
    """LAX [B,1,T,H,W], SAX [B,1,D,H,W] -> per-view class maps on CPU.

    One model call contains every frame. LAX outputs retain time order; SAX has
    one prediction conditioned on the complete observed long-axis sequences.
    """
    if model.training:
        raise ValueError('predict_cine requires model.eval()')
    if modality_mask.shape != (images['sa'].shape[0], 3):
        raise ValueError('Expected modality_mask [B,3]')
    if any(image.ndim != 5 or image.shape[1] != 1 for image in images.values()):
        raise ValueError('Expected per-view [B,1,T/D,H,W] inputs')
    outputs = model(images, modality_mask)['logits']
    predictions = {}
    for i, view in enumerate(VIEWS):
        valid = modality_mask[:, i].bool().cpu()
        if valid.any():
            pred = outputs[view].argmax(1).cpu()
            pred[~valid] = 0
            predictions[view] = pred
    return predictions
