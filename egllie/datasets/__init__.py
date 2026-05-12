from datasets.eglol import get_eglol_withNE_dataset
from datasets.egsdsd import get_egsdsd_withNE_dataset
from datasets.egreled import get_egreled_withNE_dataset
from datasets.eglie import get_lie_dataset
from os.path import join


def get_dataset(config):
    if config.NAME == "get_eglol_withNE_dataset":
        return (
            get_eglol_withNE_dataset(
                dataset_root=join(config.root, "train"),
                center_cropped_height=config.img_height,
                random_cropped_width=config.img_width,
                is_train=True,
                is_split_event=config.is_split_event,
                voxel_grid_channel=config.voxel_grid_channel
            ),
            get_eglol_withNE_dataset(
                dataset_root=join(config.root, "test"),
                center_cropped_height=config.img_height,
                random_cropped_width=config.img_width,
                is_train=False,
                is_split_event=config.is_split_event,
                voxel_grid_channel=config.voxel_grid_channel
            ),
        )
    elif config.NAME == "get_egsdsd_withNE_dataset":
        return (
            get_egsdsd_withNE_dataset(
                dataset_root=join(config.root, "train"),
                center_cropped_height=config.img_height,
                random_cropped_width=config.img_width,
                is_train=True,
                is_split_event=config.is_split_event,
                voxel_grid_channel=config.voxel_grid_channel,
                is_indoor=True
            ),
            get_egsdsd_withNE_dataset(
                dataset_root=join(config.root, "test"),
                center_cropped_height=config.img_height,
                random_cropped_width=config.img_width,
                is_train=False,
                is_split_event=config.is_split_event,
                voxel_grid_channel=config.voxel_grid_channel,
                is_indoor=True
            ),
        )
    elif config.NAME == "get_egreled_withNE_dataset":
        return (
            get_egreled_withNE_dataset(
                dataset_root=join(config.root, "train"),
                center_cropped_height=config.img_height,
                random_cropped_width=config.img_width,
                is_train=True,
                is_split_event=config.is_split_event,
                voxel_grid_channel=config.voxel_grid_channel,
            ),
            get_egreled_withNE_dataset(
                dataset_root=join(config.root, "test"),
                center_cropped_height=config.img_height,
                random_cropped_width=config.img_width,
                is_train=False,
                is_split_event=config.is_split_event,
                voxel_grid_channel=config.voxel_grid_channel,
            ),
         
        )
    elif config.NAME == "get_lie_dataset":
        return (
            get_lie_dataset(
                dataset_root=join(config.root, "train"),
                is_train=True,
                voxel_grid_channel=config.voxel_grid_channel
            ),
            get_lie_dataset(
                dataset_root=join(config.root, "test"),
                is_train=False,
                voxel_grid_channel=config.voxel_grid_channel
            ),
        )
    else:
        raise ValueError(f"Unknown dataset: {config.NAME}")
