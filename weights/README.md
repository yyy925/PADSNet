# Weight files

The code uses these relative paths by default:

```text
weights/swin_tiny_kinetics400.pth
weights/padsnet_model_epoch_100.pth
```

The first file initializes the Video Swin backbone. The second is the trained PADSNet checkpoint used by `evaluate.py`.

Weight files are excluded from Git. Publish approved weights through Git LFS, a GitHub Release, or an external model host and add the download information here.
