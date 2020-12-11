

import platform
from functools import partial

from torch.utils.data import DataLoader
from centerpoint.utils.collate import collate_kitti


if platform.system() != "Windows":
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(
    dataset, batch_size, workers_per_gpu, num_gpus=1, dist=True, **kwargs
):
    shuffle = kwargs.get("shuffle", True)

    assert dist is False

    sampler = None
    batch_size = num_gpus * batch_size
    num_workers = num_gpus * workers_per_gpu

    # TODO change pin_memory
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        collate_fn=collate_kitti,
        # pin_memory=True,
        pin_memory=False,
    )

    return data_loader