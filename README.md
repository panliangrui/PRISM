

<div align="center">
  <p>
  Liangrui Pan et al. is a developer helper.
  </p>

  <p>
    <a href="https://github.com/misitebao/yakia/blob/main/LICENSE">
      <img alt="GitHub" src="https://img.shields.io/github/license/misitebao/yakia"/>
    </a>
  </p>

  <!-- <p>
    <a href="#">Installation</a> | 
    <a href="#">Documentation</a> | 
    <a href="#">Twitter</a> | 
    <a href="https://discord.gg/zRC5BfDhEu">Discord</a>
  </p> -->

  <div>
  <strong>
  <samp>

[English](README.md)

  </samp>
  </strong>
  </div>
</div>

# Multimodal Multiscale Attention-Based Learning on Multicenter Lung Cancer CT and Histopathology Images Enhances STAS Diagnosis: A Multicenter Study

## Table of Contents

<details>
  <summary>Click me to Open/Close the directory listing</summary>

- [Table of Contents](#table-of-contents)
- [Feature Preprocessing](#Feature-Preprocessing)
- [Feature Extraction](#Feature-Extraction)
- [Models](#Train-models)
- [Test WSI](#Test WSI)
- [Datastes](#Datastes)
- [Installation](#Installation)
- [License](#license)

</details>

## Feature Preprocessing

Use the pre-trained model for feature preprocessing and build the spatial topology of WSI.
Upload the .svs file to the input folder

### Feature Extraction

Features extracted based on Prov-GigaPath and CONCHv1.5.
Please refer to Prov-GigaPath and CONCHv1.5: https://huggingface.co/prov-gigapath/prov-gigapath and https://huggingface.co/MahmoodLab/conchv1_5

Feature extraction code reference project: https://github.com/mahmoodlab/TRIDENT

**Or follow step-by-step instructions:**

**Step 1: Tissue Segmentation:** Segments tissue vs. background from a dir of WSIs
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task seg --wsi_dir ./wsis --job_dir ./trident_processed --gpu 0 --segmenter hest
   ```
   - `--task seg`: Specifies that you want to do tissue segmentation.
   - `--wsi_dir ./wsis`: Path to dir with your WSIs.
   - `--job_dir ./trident_processed`: Output dir for processed results.
   - `--gpu 0`: Uses GPU with index 0.
   - `--segmenter`: Segmentation model. Defaults to `hest`. Switch to `grandqc` for fast H&E segmentation. Add the option `--remove_artifacts` for additional artifact clean up.
 - **Outputs**:
   - WSI thumbnails in `./trident_processed/thumbnails`.
   - WSI thumbnails with tissue contours in `./trident_processed/contours`.
   - GeoJSON files containing tissue contours in `./trident_processed/contours_geojson`. These can be opened in [QuPath](https://qupath.github.io/) for editing/quality control, if necessary.

 **Step 2: Tissue Patching:** Extracts patches from segmented tissue regions at a specific magnification.
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task coords --wsi_dir ./wsis --job_dir ./trident_processed --mag 20 --patch_size 256 --overlap 0
   ```
   - `--task coords`: Specifies that you want to do patching.
   - `--wsi_dir wsis`: Path to the dir with your WSIs.
   - `--job_dir ./trident_processed`: Output dir for processed results.
   - `--mag 20`: Extracts patches at 20x magnification.
   - `--patch_size 256`: Each patch is 256x256 pixels.
   - `--overlap 0`: Patches overlap by 0 pixels (**always** an absolute number in pixels, e.g., `--overlap 128` for 50% overlap for 256x256 patches.
 - **Outputs**:
   - Patch coordinates as h5 files in `./trident_processed/20x_256px/patches`.
   - WSI thumbnails annotated with patch borders in `./trident_processed/20x_256px/visualization`.

 **Step 3a: Patch Feature Extraction:** Extracts features from tissue patches using a specified encoder
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task feat --wsi_dir ./wsis --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256 
   ```
   - `--task feat`: Specifies that you want to do feature extraction.
   - `--wsi_dir wsis`: Path to the dir with your WSIs.
   - `--job_dir ./trident_processed`: Output dir for processed results.
   - `--patch_encoder uni_v1`: Uses the `UNI` patch encoder. See below for list of supported models. 
   - `--mag 20`: Features are extracted from patches at 20x magnification.
   - `--patch_size 256`: Patches are 256x256 pixels in size.
 - **Outputs**: 
   - Features are saved as h5 files in `./trident_processed/20x_256px/features_uni_v1`. (Shape: `(n_patches, feature_dim)`)

```
```markdown
python constract_graph.py
```

## Models
**PRISM**

  <a href="(https://github.com/panliangrui/PRISM/blob/main/figure2.jpg)">
    <img src="https://github.com/panliangrui/PRISM/blob/main/figure2.jpg" width="912" height="1026" />
  </a>

Figure 2. Workflow for constructing and validating the PRISM framework using multicenter histopathological image data to predict response to targeted therapy, immunotherapy, and chemotherapy in lung cancer. a. procurement of biopsy or surgical resection specimens from patients with lung cancer; b. histopathological image acquisition and diagnostic assessment; c. digitization of glass slides into WSIs; d. WSI preprocessing. e. Schematic illustration of the PRISM framework integrating multiscale tissue representations and TME features to predict therapeutic response across targeted therapy, immunotherapy, and chemotherapy. f. Detailed architecture of the advection-diffusion graph Transformer module. g. Interpretability analysis of the multimodal model. h. Sample distribution across multicenter cohorts receiving targeted therapy, immunotherapy, and chemotherapy. 5-fold CV, internal five-fold cross-validation; IG-5CV, internal grouped five-fold cross-validation; EMCV, external multicenter validation.



**Interpretability Analysis**
```markdown
python hetmap.py
```
<a href="(https://github.com/panliangrui/PRISM/blob/main/figure5.jpg)">
    <img src="https://github.com/panliangrui/PRISM/blob/main/figure5.jpg" width="912" height="1026" />
  </a>



## Datastes

- Only features of the histopathology image data are provided as the data has a privacy protection agreement.
```markdown
Data access is available upon request via email(lip141772@gmail.com).
```
This study was designed as a retrospective, multicenter investigation conducted across four clinical centers in China between December 2020 and November 2025. The study population comprised patients with histologically confirmed lung cancer. Patients with lung adenocarcinoma underwent NGS testing and received ICI-based therapy, whereas patients with lung squamous cell carcinoma received ICI treatment without NGS-guided stratification. In total, 7,082 patients were initially screened. Pre-treatment formalin-fixed, paraffin-embedded (FFPE) whole-slide histopathological images were collected for model development and validation. After applying eligibility criteria, 1,130 patients were included in the final analysis, and 1,351 individuals were excluded for not meeting study requirements (Figure 1, Supplementary Table 1). The Second Xiangya Hospital of Central South University (SXY-CSU) cohort included patients treated between December 2020 and November 2025, comprising 343 patients receiving targeted therapy, 228 receiving immunotherapy-related regimens, and 32 receiving chemotherapy. The Xiangya Hospital of Central South University (XY-CSU) cohort enrolled patients treated between September 2022 and April 2025, including 84 patients receiving targeted therapy, 90 receiving immunotherapy, and 46 receiving chemotherapy. The Hunan Cancer Hospital (HCH) cohort included patients treated between January 2021 and January 2025, comprising 31 targeted therapy cases, 184 immunotherapy cases, and 71 chemotherapy cases. The Hunan Provincial People`s Hospital (HNNH) cohort enrolled patients treated between November 2021 and December 2025, including 6 targeted therapy cases, 28 immunotherapy cases, and 2 chemotherapy cases. Inclusion criteria were as follows: (1) age≥18 years with pathologically confirmed non-small cell lung cancer (NSCLC); (2) availability of high-quality pre-treatment FFPE whole-slide images (WSIs) suitable for digital analysis, without significant artifacts such as folding, blurring, or staining abnormalities; (3) receipt of at least one line of standard first-line clinical treatment, including targeted therapy, immunotherapy, chemotherapy, or combination regimens; and (4) complete and sufficient clinical information documented in electronic medical records, including treatment regimen and response evaluation outcomes. Exclusion criteria included incomplete clinical records or absence of histopathological images required for multiscale analysis; receipt of investigational therapies outside standard-of-care regimens prior to acquisition of baseline pathological images; presence of concurrent active malignancies other than NSCLC that could confound treatment evaluation; severe comorbidities deemed by investigators to interfere with study assessment or outcomes (such as significant cardiac, hepatic, or renal dysfunction); and pregnancy or lactation. As illustrated in Figure 2, a total of 12 cohorts derived from four centers, stratified by targeted therapy, immunotherapy, and chemotherapy, were used to develop and evaluate the PRISM model for predicting treatment response.


## Installation
- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on a single Nvidia GeForce RTX 4090)
- Python (3.7.11), h5py (2.10.0), opencv-python (4.1.2.30), PyTorch (1.10.1), torchvision (0.11.2), pytorch-lightning (1.5.10).

Note: The complete code will be made public after the paper is published.
## License
If you need the original histopathology image slides, please send a request to our email address. The email address will be announced after the paper is accepted. Thank you!

[License MIT](../LICENSE)
