today's goal:
- write a quantative test to score our model

we train a small classifier on the pixel-art sprites, then score generated
images with FID computed on the features just before its classification layer:
lower FID means the samples look like real sprites. we plot FID against the
number of sampling steps.

## get started

1. run `01_train/train.ipynb` to save `flow_mlp.pt`
2. run `test.ipynb`

homework for next time:
- either re-implement [FID](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance), or come up with a test of your choice to score your model
