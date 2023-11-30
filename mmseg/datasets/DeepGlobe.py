import os.path as osp
import tempfile

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DeepGlobe(CustomDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = ('background', 'road')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(DeepGlobe, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)

        assert osp.exists(self.img_dir)