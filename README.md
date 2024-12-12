
**FourierDiffuse: Image Diffusion in Frequency Space**
Daksh Aggarwal, Jiaying Cheng, Akshay Ghandikota

## Requirements
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ldm
```

To get started, install the additionally required python packages into your `ldm` environment
```shell script
pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
pip install git+https://github.com/arogozhnikov/einops.git
```

If you'd like to train our model, the 5 datasets (LSUN-Bedrooms, Churches, Classrooms, Conference, FFHQ) need to be obtained and placed as described in the original [LDM repo](https://github.com/CompVis/latent-diffusion).

## Training

You can choose between training the vanilla or frequency diffusion model through the model config (e.g. [here](https://github.com/dakshces/FourierImageDiffuse/blob/main/configs/latent-diffusion-frequency/lsun_churches-ldm-kl-8.yaml) and [here](https://github.com/dakshces/FourierImageDiffuse/blob/main/configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml)). Start training for unconditional image generation via:

```
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/<config_spec>.yaml -t --gpus 0,
```

## Sampling

You can sample from your saved model checkpoints using:

```
CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py -r <path_to_model_ckpt> -n <num_samples> --batch_size <bs>  -e <eta for sampling, 0.0=deterministic sampling> 
```

## FID evaluation

We carry out FID evaluation using [torch-fidelity](https://torch-fidelity.readthedocs.io/en/latest/).



Our code builds upon the original [LDM repository](https://github.com/CompVis/latent-diffusion).

## BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```


