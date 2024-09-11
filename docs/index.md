---
layout: default
---

**I have read the instructions and agree with the terms.** [Return to the application](https://hail.hope4kids.io/).

The **Brain Harmonzation Tool (HAIL)** is a free, open-source web-based application
designed at [Children's National Hospital](https://www.childrensnational.org/) 
for the harmonization of brain magnetic resonance imaging (MRI). 
Developed in Python, this software aims to provide precise quantitative analysis 
of pediatric brain MRI, to support clinical decision-making in diagnosis and prognosis.  

With its user-friendly interface, the Segmenter provides automated segmentation 
and volumetric measurements within minutes after uploading the required four MRI sequences. 
This software provides **state-of-the-art performance** powered by our benchmarked 
segmentation model, which was the **top-performing model** in the pediatric competition of the 
well-established international Brain Tumor Segmentation Challenge 
[BraTS2023](https://www.synapse.org/Synapse:syn51156910/wiki/627802).  

# Usage

This software currently functions on the the t1 MRI sequences: native pre-contrast T1-weighted (t1n) and  
contrast enhanced T1-weighted (t1c). These sequences should be 
uploaded in NIfTI format (*i.e.*, **.nii.gz**). Before uploading, 
we strongly recommend performing **de-identification** to remove any protected 
health information, including **defacing** if necessary. 

**Pre-processing** in the harmonization is under development. At this time, 
we expect users to  resample the inputs to isotropic 1 mm resolution of 192x224x192.  

Once the MRI sequences are uploaded, select the uploaded modality from the dropdown and click the **Start Harmonization** button. 
The process typically takes around 8 minutes. Afterward, the MRI 
sequence before and after harmonization is visualized on axial views 
using the interactive **Sliders**. Harmonization results can be downloaded in 
NIfTI format by clicking the **Download Harmonized File** button.  

For **demonstration** purposes, we provide sample cases at the bottom of the page. 
Select one and click the **Start Harmonization** button to see how the software works.  

# Source Code

The current version of the software is ![v1.0](https://img.shields.io/badge/v1.0-brightgreen) 
and the source code is publicly available on GitHub 
([code](https://github.com/Precision-Medical-Imaging-Group/BraTS2024-PEDS)) 
under license [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

The software is developed and maintained by the [Precision Medical Imaging](https://research.childrensnational.org/labs/precision-medical) lab
at Children’s National Hospital in Washington, DC, USA.  

# Citations

If you use and/or refer to this software in your research, please cite the following papers: 

* Z. Jiang, D. Capell&aacute;n-Mart&iacute;n, A. Parida, X. Liu, M. J. Ledesma-Carbayo, S. M. Anwar, M. G. Linguraru, 
"Enhancing Generalizability in Brain Tumor Segmentation: Model Ensemble with Adaptive Post-Processing," 
*2024 IEEE International Symposium on Biomedical Imaging (ISBI)*, Athens, Greece, 2024, pp. 1-4, 
doi: [10.1109/ISBI56570.2024.10635469](https://ieeexplore.ieee.org/document/10635469/authors#authors).

* A. Karargyris, R. Umeton, M.J. Sheller, A. Aristizabal, J. George, A. Wuest, S. Pati, et al., 
"Federated benchmarking of medical artificial intelligence with MedPerf," 
*Nature Machine Intelligence*, 5:799–810 (2023), doi: [10.1038/s42256-023-00652-2](https://doi.org/10.1038/s42256-023-00652-2).

* A. F. Kazerooni, N. Khalili, X. Liu, D. Haldar, Z. Jiang, S. M. Anwar et al., 
"The Brain Tumor Segmentation (BraTS) Challenge 2023: Focus on Pediatrics (CBTN-CONNECT-DIPGR-ASNR-MICCAI BraTS-PEDs),"
*arXiv:2305.17033 \[eess.IV\]* (2024), doi: [10.48550/arXiv.2305.17033](https://doi.org/10.48550/arXiv.2305.17033). 

* A. F. Kazerooni, N. Khalili, X. Liu, D. Gandhi, Z. Jiang, S. M. Anwar et al., 
"The Brain Tumor Segmentation in Pediatrics (BraTS-PEDs) Challenge: Focus on Pediatrics (CBTN-CONNECT-DIPGR-ASNR-MICCAI BraTS-PEDs)," 
*arXiv:2404.15009 \[cs.CV\]* (2024), doi: [10.48550/arXiv.2404.15009](https://doi.org/10.48550/arXiv.2404.15009).

# Disclaimer

* This software is provided without any warranties or liabilities and is intended for research purposes only. 
It has not been reviewed or approved for clinical use by the Food and Drug Administration (FDA) or any other federal or state agency. 

* This software does not store any information uploaded to the platform. 
All uploaded and generated data are automatically deleted once the webpage is closed. 

# Contact

* New features are continually being developed. To stay informed about updates, 
report issues or provide feedback, please fill out the 
**online form** [here](https://docs.google.com/forms/d/e/1FAIpQLSfkH4l3Dcd1qNnH_dLgbiUAQWGKeSAnbBr3ndMwD5yhoZ7Pfw/viewform) or contact [support@hope4kids.io](mailto:support@hope4kids.io) .   

* For more information, collaboration, or to receive the model in a Docker container 
for testing on large datasets locally, please contact 
**Prof. Marius George Linguraru** [contact@hope4kids.io](mailto:contact@hope4kids.io) with details on your intended use. 

**I have read the instructions and agree with the terms.** [Return to the application](https://hail.hope4kids.io/).
