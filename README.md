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

## Testing on the pre-trained network

1. Generate latent vectors for input poins:

```bash
python eval.py --pnts <path-to-point-cloud> --output <path-to-output>
```

2. Display matching results with coarse-to-fine registration:

```bash
python registration/reg_refinement.py <path-to-latents1> <path-to-latents2>
```