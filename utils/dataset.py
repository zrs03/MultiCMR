import os

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import nibabel as nib
except ImportError as error:
    raise ImportError(
        "nibabel is required for NIfTI loading. "
        "Please install via `pip install nibabel`."
    ) from error


NII_EXTENSIONS = (".nii", ".nii.gz")


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


# ============================================================
# NIfTI utilities
# ============================================================

def _load_nii(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI.

    Returns:
        data: numpy array
        affine: affine matrix
    """

    if not path:
        raise ValueError("NIfTI path is empty.")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NIfTI file not found: {path}"
        )

    nii = nib.load(path)

    data = nii.get_fdata(dtype=np.float32)
    affine = nii.affine

    return data, affine


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Z-score normalization using non-zero voxels.
    """

    non_zero = image[np.abs(image) > 1e-8]

    if non_zero.size == 0:
        return image

    mean = non_zero.mean()
    std = non_zero.std()

    if std < 1e-8:
        return image - mean

    return (image - mean) / std


def _resize_volume(
    volume: np.ndarray,
    output_size: Sequence[int],
    mode: str,
) -> np.ndarray:
    """
    Resize a 3D volume to:

        [D, H, W]

    mode:
        image -> trilinear
        label -> nearest
    """

    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume, but got shape={volume.shape}"
        )

    tensor = (
        torch.from_numpy(volume)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    if mode == "trilinear":
        resized = F.interpolate(
            tensor,
            size=tuple(output_size),
            mode="trilinear",
            align_corners=False,
        )
    elif mode == "nearest":
        resized = F.interpolate(
            tensor,
            size=tuple(output_size),
            mode="nearest",
        )
    else:
        raise ValueError(
            f"Unsupported resize mode: {mode}"
        )

    return (
        resized
        .squeeze(0)
        .squeeze(0)
        .numpy()
    )


def _random_flip_3d(
    image: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Apply the same random 3D flip to the 3-channel image
    and all corresponding masks.

    image:
        [3, D, H, W]

    masks:
        {
            "2ch": [D,H,W],
            "4ch": [D,H,W],
            "sa":  [D,H,W]
        }
    """

    for axis in range(3):

        if np.random.rand() < 0.5:

            # Image channel dimension is 0.
            # Spatial dimensions are 1,2,3.
            image = np.flip(
                image,
                axis=axis + 1,
            ).copy()

            for source in masks:
                masks[source] = np.flip(
                    masks[source],
                    axis=axis,
                ).copy()

    return image, masks


def load_cases_from_yaml(
    yaml_path: str,
    split: str = "train",
    data_root_override: str = None,
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
        "splits",
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
        "split": split,
        "num_cases": len(cases),
    }

    return cases, info


# ============================================================
# Dataset
# ============================================================

class MultiSourceNiftiDataset3D(Dataset):

    def __init__(
        self,
        cases: Sequence[CaseItem],
        output_size: Sequence[int] = (
            64,
            160,
            160,
        ),
        source_num_classes: Dict[str, int] = None,
        augment: bool = False,
        load_anno: bool = True,
    ) -> None:

        super().__init__()

        self.cases = list(cases)

        self.output_size = tuple(
            output_size
        )

        self.source_order = (
            "2ch",
            "4ch",
            "sa",
        )

        self.source_num_classes = (
            source_num_classes
            or {
                "2ch": 3,
                "4ch": 6,
                "sa": 4,
            }
        )

        self.augment = augment

        # ----------------------------------------------------
        # Important:
        #
        # train / val:
        #     load_anno=True
        #
        # test:
        #     load_anno=False
        # ----------------------------------------------------

        self.load_anno = load_anno

    def __len__(self) -> int:

        return len(self.cases)

    def __getitem__(self, index: int):

        item = self.cases[index]

        # ====================================================
        # 1. Load 2CH / 4CH / SAX images
        # ====================================================

        images = []

        affine = None

        original_shapes = {}

        for source in self.source_order:

            image_path = item.image_paths[source]

            image, current_affine = _load_nii(
                image_path
            )

            if affine is None:
                affine = current_affine

            original_shapes[source] = tuple(
                image.shape
            )

            # --------------------------------------------
            # Image normalization
            # --------------------------------------------

            image = _normalize_image(
                image
            )

            # --------------------------------------------
            # Resize image
            # --------------------------------------------

            image = _resize_volume(
                image,
                self.output_size,
                mode="trilinear",
            )

            images.append(image)

        # ====================================================
        # 2. Stack three images into 3 channels
        # ====================================================

        # Each image:
        #
        #     [D,H,W]
        #
        # After stack:
        #
        #     [3,D,H,W]
        #

        image = np.stack(
            images,
            axis=0,
        ).astype(
            np.float32
        )

        # ====================================================
        # 3. Load annotations
        # ====================================================

        masks = {}

        if self.load_anno:

            for source in self.source_order:

                if source not in item.label_paths:
                    raise ValueError(
                        f"Annotation path missing for "
                        f"case={item.case_id}, "
                        f"source={source}"
                    )

                label_path = item.label_paths[
                    source
                ]

                label, _ = _load_nii(
                    label_path
                )

                # ----------------------------------------
                # Convert to integer class labels
                # ----------------------------------------

                label = np.rint(
                    label
                ).astype(
                    np.int64
                )

                expected_classes = (
                    self.source_num_classes.get(
                        source
                    )
                )

                if expected_classes is None:
                    raise KeyError(
                        f"Missing class count for "
                        f"source={source}"
                    )

                # ----------------------------------------
                # Check label range BEFORE resize
                # ----------------------------------------

                valid_min = int(
                    label.min()
                )

                valid_max = int(
                    label.max()
                )

                if (
                    valid_min < 0
                    or valid_max >= expected_classes
                ):
                    raise ValueError(
                        f"Label range out of bounds "
                        f"for source={source}, "
                        f"case={item.case_id}. "
                        f"Got [{valid_min}, {valid_max}] "
                        f"but expected "
                        f"[0, {expected_classes - 1}]"
                    )

                # ----------------------------------------
                # Resize label
                #
                # VERY IMPORTANT:
                # segmentation uses nearest interpolation.
                # ----------------------------------------

                label = _resize_volume(
                    label.astype(
                        np.float32
                    ),
                    self.output_size,
                    mode="nearest",
                ).astype(
                    np.int64
                )

                masks[source] = label

        # ====================================================
        # 4. Data augmentation
        # ====================================================

        if self.augment:

            image, masks = _random_flip_3d(
                image,
                masks,
            )

        # ====================================================
        # 5. Convert image to Tensor
        # ====================================================

        image_tensor = torch.from_numpy(
            image
        ).float()

        # Shape:
        #
        #     [3,D,H,W]
        #

        # ====================================================
        # 6. Convert masks to Tensor
        # ====================================================

        result = {
            "image": image_tensor,

            "case_id": item.case_id,

            "image_paths": item.image_paths,

            "affine": torch.from_numpy(
                affine
            ).float(),

            "original_shapes": original_shapes,

            "has_anno": self.load_anno,
        }

        # ====================================================
        # Train / Val
        # ====================================================

        if self.load_anno:

            # Individual masks:
            #
            # masks["2ch"] -> [D,H,W]
            # masks["4ch"] -> [D,H,W]
            # masks["sa"]  -> [D,H,W]

            result["masks"] = {
                source: torch.from_numpy(
                    masks[source]
                ).long()
                for source in self.source_order
            }

            # Also provide a stacked version:
            #
            # [3,D,H,W]
            #
            result["mask"] = torch.stack(
                [
                    torch.from_numpy(
                        masks[source]
                    ).long()
                    for source in self.source_order
                ],
                dim=0,
            )

            result["label_paths"] = (
                item.label_paths
            )

        # ====================================================
        # Test
        # ====================================================

        else:

            # Test has no annotation.
            #
            # Do NOT create fake masks.

            result["masks"] = None
            result["mask"] = None
            result["label_paths"] = {}

        return result