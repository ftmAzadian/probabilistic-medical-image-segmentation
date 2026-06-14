# segmentation_core.py
import nibabel as nib
import numpy as np
from skimage.transform import resize
from .tissue_segmenter import TissueSegmenter

def loading_templates(mri_data, csf_template_path, scalp_template_path, skull_template_path):
    templates = {}
    for tissue, path in {'csf': csf_template_path, 'scalp': scalp_template_path, 'skull': skull_template_path}.items():
        template = nib.load(path).get_fdata()
        template_resized = resize(template, mri_data.shape, mode='constant', anti_aliasing=True)
        template_resized = (template_resized - np.min(template_resized)) / (np.max(template_resized) - np.min(template_resized) + 1e-6)
        templates[tissue] = template_resized.astype(np.float32)
    return templates

def run_segmentation_pipeline(mri_data, csf_template_path, scalp_template_path, skull_template_path):
    templates = loading_templates(mri_data, csf_template_path, scalp_template_path, skull_template_path)
    segmenter = TissueSegmenter()

    mri_norm = segmenter.preprocess_mri(mri_data)
    segmenter.compute_tissue_thresholds(templates)
    segmenter.compute_tissue_weights(mri_norm, templates)

    mri_enhanced = mri_norm.copy()
    for tissue in ['csf', 'scalp', 'skull']:
        mri_enhanced = segmenter.enhance_tissue(mri_enhanced, templates[tissue], tissue)

    segmentation, mri_norm = segmenter.segment_tissues(mri_norm, mri_enhanced, templates)
    final_result = segmenter.post_process_segmentation(segmentation, mri_norm)

    return final_result
