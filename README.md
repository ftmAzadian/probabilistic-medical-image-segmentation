# Probabilistic Medical Image Segmentation (Neonatal MRI)

## Overview

This project implements a neonatal brain MRI segmentation system using probabilistic atlases and Markov Random Field (MRF) modeling.

It is integrated as a research plugin into the MedVisPy medical imaging platform and is designed for segmenting brain tissues including:
- Skull  
- Scalp  
- Cerebrospinal Fluid (CSF)  
- Background  

---

## Research Problem

Neonatal MRI segmentation is challenging due to:

- Poor visibility of skull structures in MRI
- High anatomical variability in newborn brains
- Safety limitations of CT imaging (radiation exposure)

This project addresses these issues using probabilistic modeling instead of relying on CT-based priors alone.

---

## Methodology

The proposed framework combines:

- Probabilistic atlas construction from prior MRI/CT data
- Markov Random Field (MRF) spatial modeling
- Expectation-Maximization (EM) optimization
- Statistical + spatial feature fusion

---

## System Architecture

Input MRI Volume  
→ Probabilistic Atlas Initialization  
→ MRF-Based Inference  
→ EM Optimization  
→ Final Tissue Segmentation Output  

---

## Implementation

- Python 3.10+
- NumPy (vectorized computation)
- Numba (performance optimization)
- SciPy and NiBabel for medical image processing
- MedVisPy plugin integration

---

## Repository Structure

- `plugin_medvispy/` → Core segmentation algorithm and MedVisPy integration layer  
- `evaluation/` → Evaluation scripts and accuracy metrics  
- `results/` → Sample segmentation outputs  

---

## Key Contributions

- Probabilistic medical image segmentation framework
- MRF-based spatial modeling for neonatal MRI
- Efficient implementation using Python optimization techniques
- Integration into a real medical imaging research platform (MedVisPy)

---

## Results

The system was evaluated on neonatal MRI datasets and compared with standard tools:

- BET (FSL)
- FAST (FSL)

### Performance:
- Processing time: < 10 minutes per MRI volume  
- Competitive segmentation accuracy  
- Robust performance across multiple test cases  

---

## Author

Fatemeh Azadian  
K. N. Toosi University of Technology
