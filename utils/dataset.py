import os

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocess.processing import VIEWS, load_view, validate_output_sizes
from .augmentation import random_spatial_affine

# ============================================================
# Case definition
# ============================================================

@dataclass
class CaseItem:
    """
    One matched case containing 2CH / 4CH / SAX.

    image_paths:
        {
            "2ch": "...",
            "4ch": "...",
            "sa": "..."
        }

    label_paths:
        Train/Val:
            {
                "2ch": "...",
                "4ch": "...",
                "sa": "..."
            }

        Test:
            {}
    """

    case_id: str
    image_paths: Dict[str, str]
    label_paths: Dict[str, str]
    sample_id: str = ""


def load_cases_from_yaml(
    yaml_path: str,
    split: str = "train",
    data_root_override: str = None,
    complete_cine: bool = False,
) -> Tuple[List[CaseItem], dict]:
    """
    Load matched 2CH / 4CH / SAX cases from YAML.

    Path handling:
        1. Absolute path:
           Use directly.

        2. Relative path:
           Join with dataset_info.data_root.

    Expected YAML structure:

        dataset_info:
          data_root: /home/.../rawdata/CINE_MULTI
          source_order:
            - 2ch
            - 4ch
            - sa

        splits:
          train:
            - case_id: "084"
              2ch:
                image: /home/.../2CH_TR/image/CINE_2CH_084.nii.gz
                anno: /home/.../2CH_TR/anno/CINE_2CH_084.nii.gz
              4ch:
                image: /home/.../4CH_TR/image/CINE_4CH_084.nii.gz
                anno: /home/.../4CH_TR/anno/CINE_4CH_084.nii.gz
              sa:
                image: /home/.../SAX_TR/image/CINE_SAX_084.nii.gz
                anno: /home/.../SAX_TR/anno/CINE_SAX_084.nii.gz

          test:
            - case_id: "084"
              2ch:
                image: /home/.../2CH_TST/image/CINE_2CH_084.nii.gz
              4ch:
                image: /home/.../4CH_TST/image/CINE_4CH_084.nii.gz
              sa:
                image: /home/.../SAX_TST/image/CINE_SAX_084.nii.gz
    """

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Dataset split YAML not found: {yaml_path}"
        )

    with open(
        yaml_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = yaml.safe_load(f)

    dataset_info = data.get(
        "dataset_info",
        {},
    )

    if dataset_info.get('schema') != 'sax_context_v2':
        raise ValueError('Old cine manifest: rerun bash run.sh preprocess for SAX slice/phase context samples')

    # --------------------------------------------------------
    # Resolve data root
    # --------------------------------------------------------
    root = (
        data_root_override
        or dataset_info.get(
            "data_root",
            "",
        )
    )

    if root:
        root = os.path.abspath(root)

    splits_dict = data.get(
        "case_splits" if complete_cine else "splits",
        {},
    )

    # --------------------------------------------------------
    # Select split
    # --------------------------------------------------------
    if split == "all":
        records = (
            splits_dict.get("train", [])
            + splits_dict.get("val", [])
            + splits_dict.get("test", [])
        )
    elif split in splits_dict:
        records = splits_dict[split]
    else:
        raise KeyError(
            f"Split {split} not found in YAML. "
            f"Available keys: {list(splits_dict.keys())}"
        )

    # --------------------------------------------------------
    # Path resolver
    # --------------------------------------------------------
    def resolve_path(path: str) -> str:
        if not path:
            return ""

        # Absolute path:
        # use it directly.
        if os.path.isabs(path):
            return os.path.abspath(path)

        # Relative path:
        # join with data_root.
        if not root:
            raise ValueError(
                f"Relative path encountered but data_root is empty: "
                f"{path}"
            )

        return os.path.abspath(
            os.path.join(root, path)
        )

    # --------------------------------------------------------
    # Convert records to CaseItem
    # --------------------------------------------------------
    cases = []

    source_order = dataset_info.get(
        "source_order",
        ["2ch", "4ch", "sa"],
    )

    for rec in records:

        case_id = str(
            rec.get("case_id", "")
        )

        image_paths = {}
        label_paths = {}

        for source in source_order:

            if source not in rec:
                raise KeyError(
                    f"Case {case_id} is missing source={source}"
                )

            source_record = rec[source]

            # ------------------------------------------------
            # Image
            # ------------------------------------------------
            img_p = source_record.get(
                "image",
                "",
            )

            if not img_p:
                raise ValueError(
                    f"Case {case_id}, source={source} "
                    f"has no image path."
                )

            img_p = resolve_path(img_p)

            image_paths[source] = img_p

            # ------------------------------------------------
            # Annotation
            # ------------------------------------------------
            lbl_p = source_record.get(
                "anno",
                "",
            )

            if lbl_p:
                lbl_p = resolve_path(lbl_p)
                label_paths[source] = lbl_p

        cases.append(
            CaseItem(
                case_id=case_id,
                image_paths=image_paths,
                label_paths=label_paths,
                sample_id=str(rec.get("sample_id", case_id)),
            )
        )

    info = {
        "source_order": source_order,
        "num_classes_by_source": dataset_info.get(
            "num_classes_by_source",
            {
                "2ch": 3,
                "4ch": 6,
                "sa": 4,
            },
        ),
        "data_root": root,
        "output_sizes": dataset_info.get("output_sizes"),
        "schema": dataset_info["schema"],
        "split": split,
        "num_cases": len(cases),
    }

    return cases, info


# ============================================================
# Dataset
# ============================================================

class MultiSourceNiftiDataset3D(Dataset):
    """LAX [1,T,H,W] and SAX [1,D,H,W], with complete per-view labels.

    Training/validation use cached windows and center labels. complete_cine is for reconstruction at test time.
    The historical class name is retained for existing imports.
    """

    def __init__(self, cases: Sequence[CaseItem], output_sizes=None,
                 source_num_classes: Dict[str, int] = None, augment: bool = False,
                 load_anno: bool = True, complete_cine: bool = False,
                 affine_prob=.5, rotation_degrees=10., translation_fraction=.05, scale_fraction=.1):
        if not (0 <= affine_prob <= 1 and 0 <= rotation_degrees <= 180 and
                0 <= translation_fraction < 1 and 0 <= scale_fraction < 1):
            raise ValueError('Invalid affine augmentation bounds')
        self.affine_config = dict(probability=affine_prob, degrees=rotation_degrees,
                                  translate=translation_fraction, scale=scale_fraction)
        self.cases = list(cases)
        self.complete_cine = complete_cine
        self.output_sizes = validate_output_sizes(output_sizes)
        self.source_order = VIEWS
        self.source_num_classes = source_num_classes or {'2ch': 3, '4ch': 6, 'sa': 4}
        self.augment, self.load_anno = augment, load_anno

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        item = self.cases[index]
        images, masks, affines, original_shapes = {}, {}, {}, {}
        for source in self.source_order:
            size = self.output_sizes[source]
            image, affine, original_shape = load_view(item.image_paths[source], source, size, complete_cine=self.complete_cine)
            affines[source] = torch.as_tensor(affine.copy(), dtype=torch.float32)
            original_shapes[source] = tuple(int(n) for n in original_shape)
            label = None
            if self.load_anno:
                if source not in item.label_paths:
                    raise ValueError(f'Annotation missing for case={item.case_id}, source={source}')
                label, _, label_shape = load_view(item.label_paths[source], source, size, is_label=True, complete_cine=self.complete_cine)
                if tuple(label_shape) != original_shapes[source]:
                    raise ValueError(f'Image/label shape mismatch: case={item.case_id}, source={source}')
                if label.min() < 0 or label.max() >= self.source_num_classes[source]:
                    raise ValueError(f'Label range out of bounds: case={item.case_id}, source={source}')
            if self.augment:
                # Flip spatial axes only: never reverse the LAX temporal axis.
                axes = (image.ndim - 2, image.ndim - 1)
                for axis in axes:
                    if np.random.rand() < .5:
                        image = np.flip(image, axis=axis)
                        if label is not None:
                            label_axis = axis - 1 if not self.complete_cine else axis
                            label = np.flip(label, axis=label_axis)
                image, label = random_spatial_affine(image, label, **self.affine_config)
            images[source] = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)
            if label is not None:
                masks[source] = torch.from_numpy(np.ascontiguousarray(label)).long()
        with np.load(item.image_paths['sa'], allow_pickle=False) as sax_cache:
            sax_phases = int(sax_cache['sax_phases'])
        return {
            'sax_phases': sax_phases,
            'image': images, 'masks': masks, 'case_id': item.case_id, 'sample_id': item.sample_id,
            'image_paths': item.image_paths, 'label_paths': item.label_paths,
            'affines': affines, 'original_shapes': original_shapes,
            'has_anno': self.load_anno,
        }
