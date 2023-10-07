# ImplicitReg

Code accompanying the paper "Learning Invariant Implicit Shape Features from RGB-D Videos for Efficient Reconstruction and Relocalization (under review)"

## Dependencies

Code tested in Ubuntu22.04 + CUDA11.1. The following libraries/packages are required to run the code:

+ PyTorch >= 1.9
+ Pillow
+ OpenCV
+ Open3D
+ trimesh
+ pyrender
+ scipy

## Datasets used in the experiments

All data used in this paper are publicly available. Training data ShapeNet can be found at [https://shapenet.org/]. Regarding reconstruction experiments, ICL-NUIM, Scene3D and ScanNet can be found at [https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html], [https://qianyi.info/scenedata.html] and [http://www.scan-net.org/]. Finally, for the relocalization task, the 7 Scenes dataset is available at [https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/].

## Testing on the pre-trained network

1. Generate latent vectors for input poins:

```bash
python eval.py --pnts <path-to-point-cloud> --output <path-to-output>
```

2. Display matching results with coarse-to-fine registration:

```bash
python registration/reg_refinement.py <path-to-latents1> <path-to-latents2>
```
