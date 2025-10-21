# LibreFLUX-ControlNet
![LibreFLUX example](examples/side_by_side_b.png)

The intent of this REPO is to make it possible to train a controlnet based on LibreFLUX. This means:
- Incorperating Attention Masking
- Removing the distilled guidance vector during training
- Running inference with CFG.

**Disclaimer**, This is pieced together by modifying and borrowing from these repos: 
- https://github.com/christopher-beckham/flux-controlnet ( Its a fork of this )
- https://huggingface.co/jimmycarter/LibreFLUX
- https://github.com/bghira/SimpleTuner

Use at your own risk!

## Setup

### Environment

Create a conda environment:

```
conda create -n <my env name> python=3.11
conda activate <my env name>
```
### Setup

- Clone the repository.
- Change the current directory to `LibreFLUX-ControlNet/`.
- Install the required dependencies using the `requirements.txt` file.
```
git clone https://github.com/NeuralVFX/LibreFLUX-ControlNet/
cd LibreFLUX-ControlNet/
pip install -r requirements.txt
```


### Dataset

I've prepared a simple dataset in `sam_dataset` so you can see how to structure one. I assume that the dataset will have both the image, and the control, in seperate folders. The image directory should contain txt files which hold captions for each image.

### Configuration

Firstly, cd into `exps`, and modify `env.sh` to make sure the following variables are defined:
- `$DATA_DIR`: where your dataset is located, e.g. `/sam_dataset/sam_img/`.
- `$SAVE_DIR`: where to save experiments, e.g. `./save_dir`.
- (optional) `NEPTUNE_API_KEY` for logging to Neptune. If you use Neptune, make sure to install it with `pip install neptune`.

Also, if you are dealing with gated models or want to upload them then ensure you are authenticated on huggingface via `huggingface-cli login`.

## Training

Please see `exps/train_libre.sh` for an example script. The script usage is as follows:

```
cd exps
source env.sh # do this just once, very important
bash train_libre.sh <name of experiment>
```

For instance, running with `bash train_libre.sh my_experiment` gives the following printout before the rest of the code is executed:

```
-------------------------------
Exp group        :   my_experiment_123
Exp id           :   1727111804
Experiment name  :   my_experiment_123/1727111804
Output directory :   /home/chris/results/flux-adapters/my_experiment_123/1727111804
Data dir         :   /home/chris/datasets/dog-example
-------------------------------
```

i.e., `my_experiment_123/1727111804` is the unique identifier of the experiment which lives under `$SAVE_DIR`. Currently, the digit identifier is just Unix time, but you can modify this to support whatever experiment scheduler system you're using (e.g. Slurm). Note that if you invoke the script like so:

For quicker dataset discovery, set arg:

`--num_workers 32`: Allows multi-threaded dataset discovery


```
bash train_libre.sh <experiment name>/<id>
```

then `<id>` will be used instead as the digit identifier. If you use Neptune to log experiments, then you can run `NEPTUNE_PROJECT=<neptune project name> bash train.sh <experiment name>` instead. 

It _should_ be possible to train on a single 40GB A100 GPU for a "modestly"-sized ControlNet. By default, the training script sets `args.num_layers=2` and `args.num_single_layers=4` (i.e. "depth 2-4"). For reference, the Union ControlNet-Pro by Shakker Labs (`Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro`) uses depth 5-10 and probably won't run, but if you want to see how you could finetune on top of it then you can look at `exps/train-510.sh`.

Here is a breakdown of how we save on GPU memory:

- `--offload_gpu 1`: moves the VAE to the second GPU ( Added by NeuralVFX )
- `--gradient_checkpointing`: save memory by recomputing activations during backward instead of storing them.
- `--mixed_precision bf16`: everything gets cast into this precision, even the ControlNet.
- `--use_8bit_adam`: ADAM uses a lot of GPU memory since it has to store statistics for each parameter being trained. Here we use the 8-bit version.
- `--quantize`: quantise everything (except ControlNet, text encoders, and certain layers of the transformer) into `int8` via the `optimum-quanto` library. This is _weight only_ quantisation, so params are stored in `int8` and are de-quantised on the fly. You may be able to squeeze out even more savings with lower bits but this has not been tested.

### Debugging and hyperparameters

Some things to consider:
- To increase the effective batch size, you can increase `gradient_accumulation_steps` but this will slow down training by that same factor.
- You should not make the number of data loader workers too large. This is because each data loader has a copy of the line-art detector which adds to the overall GPU memory. If this is problematic, then you should preprocess the dataset and modify `dataset.py` to also load the conditioning images in.
- To bring down GPU usage even more, you could precompute the conditioning images and modify `metadata.jsonl` and `dataset.py` accordingly to directly load them in. Furthermore, you can also squeeze some savings out by precomputing the VAE and text encoder embeddings.

Various "quality of life" printouts are done during the execution of the script to make sure everything works as intended. Most notably this:

```
name            dtypes            devices                            # params    # params L
--------------  ----------------  ------------------------------  -----------  ------------
vae             {torch.bfloat16}  {device(type='cuda', index=0)}  8.38197e+07   0
text_encoder    {torch.bfloat16}  {device(type='cuda', index=0)}  1.2306e+08    0
text_encoder_2  {torch.bfloat16}  {device(type='cuda', index=0)}  4.76231e+09   0
transformer     {torch.bfloat16}  {device(type='cuda', index=0)}  1.19014e+10   0
controlnet      {torch.bfloat16}  {device(type='cuda', index=0)}  1.34792e+09   1.34792e+09
```

For each pipeline component we scan through all the parameters and list _all_ the dtypes we found*, and _all_ the devices they were found on. We also count the total number of params and learnable parameters, respectively. (* One exception however is that even with `--quantize` enabled we won't see the actual internal dtype of the weights (which should be `qint8`), this is due to a certain abstraction implemented in `optimum-quanto` which makes it so that quanto tensors "look like" `bf16` even though the internal representation is actually int.)

As you can see, under the default arguments the ControlNet is ~1,347,920,000 (~1.3B) params which is absolutely ginormous in absolute scale but still "small" in relative scale (the base transformer is ~11B params, so this amounts to ~10% of that size). From personal experience it seems as though this model size is "required" to get a decent performing model (short of a more long term solution in finding a more efficient architecture). For reference, the Shakker Labs CN-Union-Pro is much larger and sits at ~3.3B params. (While you can try to run this in `exps/train-510.sh`, it will probably fail.)

## Inference Example: Single Image + Prompt

Use the simple mode to generate from one control image and one prompt, no metadata.jsonl required.

```python inference.py \
  --base_model "jimmycarter/LibreFlux-SimpleTuner" \
  --controlnet_path "/content/drive/MyDrive/pod_output/libreflux_controlnet_1024/checkpoint-165000/controlnet-cuda" \
  --output_dir "/content/LibreFLUX-ControlNet/inference_test" \
  --steps 75 \
  --image "/content/LibreFLUX-ControlNet/test_cc/libre_flux.png" \
  --prompt "many pieces of drift wood spelling libre flux sitting casting shadow on the lumpy sandy beach with foot prints all over it"
```

Notes:

```
--prompt is the positive text prompt.
```

Optional quality controls: 
```
--guidance_scale 4.0
--controlnet_conditioning_scale 1.0
--negative_prompt "blurry, bokeh, jpeg artifacts"
```

Batch mode with JSONL still works the same; use `--metadata_dir` instead of `--image` and `--prompt`

## Uploading to the Hub

Simply cd into `exps` and run:

```
bash upload.sh <experiment name>/checkpoint-<iteration> <org>/<model name> --model_type=controlnet
```

Note that `<experiment name>/checkpoint-<iteration>` is relative to `$SAVE_DIR`. `<org>` will be either your HuggingFace username or organisation name.


