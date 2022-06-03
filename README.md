# Submission entry for MABE_2020 competition

The method primiarly uses permuation invariant features to create a per frame embedding of mouse triplet keypoint data


# File structure


```
helper
---pointnet (pointnet related files)
--features.py (functions for creating hand crafted features)
--pca_feat.py (performing pose PCA on keypoints)
--utils.py (general purpose functions)
generate_sample_submission.py ( creates final embedding)
```