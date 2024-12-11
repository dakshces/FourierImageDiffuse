import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/class2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()

    print("AFTER PARSINGGGGG")

    config = OmegaConf.load("configs/latent-diffusion-frequency/cin256_added.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, "logs/2024-12-09T06-52-45_cin256_added/checkpoints/epoch=000019.ckpt")
    
    print("after load modelLLLL")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir



    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()

    #classes = [25, 187, 448, 992]   # define classes to be sampled here
    #classes = [50, 900, 200, 400]
    #classes = [100, 300, 500, 800]
    classes = [0, 217, 497, 574]  
    #imagenette total classes:
    #0 (man holding fish), 217  (dog) , 482 (n02979186) (cassete player/radio,etc) , 
    #491: n03000684 (chainsaw in setting/solo)  497: n03028079 (churches/cathedral/monuments)
    # 566: n03394916 (trumpet/horn/band/instruments)  569: n03417042 (garbage truck), 571: n03425413 (gas station)
    # 574: n03445777 (golfball/golfing) , 701: n03888257(parachute flying)



    with torch.no_grad():
        with model.ema_scope():
            print("before get learned conditiong")
            
            #uc = model.get_learned_conditioning(
            #    {model.cond_stage_key: torch.tensor(opt.n_samples*[1000]).to(model.device)}
            #    )
            
            for class_label in classes:
                print(f"rendering {opt.n_samples} examples of class '{class_label}' in {opt.ddim_steps} steps and using s={opt.scale:.2f}.")
                xc = torch.tensor(opt.n_samples*[class_label])
                print("before get learned conditioning")
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                print("gpu", c.device)
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                conditioning=c,
                                                batch_size=opt.n_samples,
                                                shape=[3, opt.H//4, opt.W//4],
                                                verbose=False,
                                                #unconditional_guidance_scale=opt.scale,
                                                #unconditional_conditioning=uc, 
                                                eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)



    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'class2img.png'))

    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")

# python scripts/class2img.py  --ddim_eta 0.0 --n_samples 6  --scale 3.0  --ddim_steps 20