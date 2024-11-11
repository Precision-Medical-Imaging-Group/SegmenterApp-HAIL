---
layout: default
---

**I have read the instructions and agree with the terms.** [Return to the application](https://hail.hope4kids.io/).

The **Brain Harmonization Tool (HAIL)** is a free, open-source web-based application
designed at [Children's National Hospital](https://www.childrensnational.org/) 
for the harmonization of brain magnetic resonance imaging (MRI). 
Developed in Python, this software aims to be used as a preprocssing tool to provide precise quantitative analysis of brain MRI and support clinical decision-making in diagnosis and prognosis.  

# Usage

This software currently functions on the t1 MRI sequences: native pre-contrast T1-weighted (t1n) and  
contrast-enhanced T1-weighted (t1c). These sequences should be 
uploaded in NIfTI format (*i.e.*, **.nii.gz**). Before uploading, 
we strongly recommend performing **de-identification** to remove any protected 
health information, including **defacing** if necessary. 

**Pre-processing** in the harmonization is under development. At this time, 
we expect users to  resample the inputs to an isotropic 1 mm resolution of 192x224x192.  

Once the MRI sequences are uploaded, select the uploaded modality from the dropdown and click the **Start Harmonization** button. 
The process typically takes around 8 minutes. Afterward, the MRI 
sequence before and after harmonization is visualized on axial, coronal, and sagittal views using the interactive **Sliders**. Harmonization results can be downloaded in 
NIfTI format by clicking the **Download Harmonized File** button.  

For **demonstration** purposes, we provide sample cases at the bottom of the page. 
Select one and click the **Start Harmonization** button to see how the software works.  

# Source Code

The current version of the software is![v1.0](https://img.shields.io/badge/v1.0-brightgreen) 
and the source code is publicly available on GitHub 
([code](https://github.com/Precision-Medical-Imaging-Group/SegmenterApp-HAIL) and [docs](https://docs.hope4kids.io/SegmenterApp-HAIL/hail.html))
under license [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

The software is developed and maintained by the [Precision Medical Imaging](https://research.childrensnational.org/labs/precision-medical) lab
at Children’s National Hospital in Washington, DC, USA.  

# Citations

If you use and/or refer to this software in your research, please cite the following papers: 

* Parida, Abhijeet, Zhifan Jiang, Syed Muhammad Anwar, Nicholas Foreman, Nicholas Stence, Michael J. Fisher, Roger J. Packer, Robert A. Avery, and Marius George Linguraru. "Harmonization Across Imaging Locations (HAIL): One-Shot Learning for Brain MRI." arXiv preprint arXiv:2308.11047 (2023).

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
