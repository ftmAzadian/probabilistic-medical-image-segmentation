from Plugins_Manager.Plugin_Collection import Plugin
import numpy as np
from .segmentation_core import run_segmentation_pipeline

class CSFSkullScalpSegmentation(Plugin):

    def __init__(self):
        super().__init__()
        self.description = 'Tissue segmentation plugin'
        self.type = 'TissueSegmentation'

    def perform_operation(self, argument):
        mri_data = argument['mri_data']
        csf_template_path = argument['csf_template']
        scalp_template_path = argument['scalp_template']
        skull_template_path = argument['skull_template']

        result = run_segmentation_pipeline(mri_data, csf_template_path, scalp_template_path, skull_template_path)
        result = (result/255.0)*3
        return [{
            'segmentation_result': result.astype(np.uint8),
            'label_names': ['Background', 'CSF', 'Scalp', 'Skull'],
            'label_values': [0, 64, 175, 255]
        }]
