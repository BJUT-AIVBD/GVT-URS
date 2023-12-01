import os.path as osp
import tempfile

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class UnpavedRoad(CustomDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = ('background', 'unpaved_road', 'paved_road')

    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0]]

    def __init__(self, **kwargs):
        super(UnpavedRoad, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)

        assert osp.exists(self.img_dir)