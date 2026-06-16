today's goals:
- train our first image model and generate images.
- warm up, get a sense of how easy it can be to train useful/fun models

homework for next time:
- find a dataset, write the loading code and train your own model

## get started locally

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                    # install uv
uv sync                                                            # install deps into .venv
uv run python -m ipykernel install --user --name ai-image-models   # register the kernel
uv run jupyter lab                                                 # open the notebooks
```

then pick the **ai-image-models** kernel in jupyter.

## get started in the cloud

upload train.ipynb to colab/modal
