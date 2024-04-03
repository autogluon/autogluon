# Conv-LoRA: Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model (ICLR 2024)

Examples showing how to use `Conv-LoRA` for parameter efficient fine-tuning SAM.

## 1. Installation
The installation may take a while since AutoGluon Multimodal has multiple dependencies.
```shell
  conda create -n conv-lora python=3.10
  conda activate conv-lora
  pip install -U pip
  pip install -U setuptools wheel
  git clone https://github.com/autogluon/autogluon
  cd autogluon && pip install -e multimodal/[tests]
  ```

## 2. Dataset

Enter the `autogluon/examples/automm/Conv-LoRA` directory and run the following script to download the datasets.

`python prepare_semantic_segmentation_datasets.py`

## 3. Training

`python run_semantic_segmentation.py --<flag> <value>`

- `task` refers to the dataset name, i.e., one of the datasets we have downloaded. Options are `polyp, leaf_disease_segmentation, camo_sem_seg, isic2017, road_segmentation, or SBU-shadow`.
- `seed` determines the random seed.
- `rank` determines the rank of Conv-LoRA. Default is 3.
- `expert_num` determines the used expert number of Conv-LoRA. Default is 8.
- `num_gpus` determines the number of gpu used for training. Default is 1.
- `output_dir` determines the path of output directory. Default is "outputs" folder.
- `ckpt_path` determines the path of model for evaluation. Default is "outputs" folder.
- `per_gpu_batch_size` is the batch size for each GPU. Default is 1.
- `batch_size` effective batch size. If batch_size > per_gpu_batch_size * num_gpus, gradient accumulation would be used. Default is 4.

## 4. Evaluation

After running the benchmark, the evaluation results of test set are stored in "{output_dir}/metrics.txt".

You can also run the following command to evaluate a checkpoint:

`python3 run_semantic_segmentation.py --task {dataset_name} --output_dir {output_dir} --ckpt_path {ckpt_path} --eval`


### Citation

```
@article{zhong2024convolution,
  title={Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model},
  author={Zhong, Zihan and Tang, Zhiqiang and He, Tong and Fang, Haoyang and Yuan, Chun},
  journal={arXiv preprint arXiv:2401.17868},
  year={2024}
}
```