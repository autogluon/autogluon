import argparse
from autogluon.text.automm import AutoMMPredictor
from datasets import load_dataset

from time import time

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

GLUE_METRICS = {
    "mnli": {"val": "accuracy", "eval": "accuracy"},
    "qqp": {"val": "accuracy", "eval": ["accuracy", "f1"]},
    "qnli": {"val": "accuracy", "eval": "accuracy"},
    "sst2": {"val": "accuracy", "eval": "accuracy"},
    "stsb": {"val": "pearsonr", "eval": ["pearsonr", "spearmanr"]},  # P/S correlation?
    "mrpc": {"val": "accuracy", "eval": "accuracy"},
    "rte": {"val": "accuracy", "eval": "accuracy"},
    # "cola": "", #phi coeffiecient is not implemented
}


def main(args):
    assert args.glue_task in (list(GLUE_METRICS.keys()) + ["mnlim", "mnlimm"]), 'Unsupported dataset name.'

    ### Dataset Loading
    if args.glue_task == "mnlimm":
        glue_task = "mnli"
        mnli_mismatched = True
    elif args.glue_task in ["mnli" or "mnlim"]:
        glue_task = "mnli"
        mnli_mismatched = False
    else:
        glue_task = args.glue_task
    dataset = load_dataset("glue", glue_task)
    train_df = dataset["train"].to_pandas().drop("idx", axis=1)
    if args.glue_task == "mnli":
        if mnli_mismatched:
            valid_df = dataset["validation_matched"].to_pandas()
        else:
            valid_df = dataset["validation_mismatched"].to_pandas()
    else:
        valid_df = dataset["validation"].to_pandas()

    ### Train and evaluate the teacher model
    teacher_predictor = AutoMMPredictor(label="label", eval_metric=GLUE_METRICS[glue_task]["val"])
    teacher_predictor.fit(
        train_df,
        hyperparameters={
            "env.num_gpus": args.num_gpu,
            "model.hf_text.checkpoint_name": args.teacher_model,
            "optimization.learning_rate": 1.0e-4,
            "optimization.weight_decay": 1.0e-3,
        },
        time_limit=args.time_limit,
        seed=args.seed,
    )
    start = time()
    teacher_result = teacher_predictor.evaluate(data=valid_df, metrics=GLUE_METRICS[glue_task]["eval"])
    teacher_usedtime = time() - start

    ### Train and evaluate a smaller pretrained model
    pretrained_predictor = AutoMMPredictor(label="label", eval_metric=GLUE_METRICS[glue_task]["val"])
    pretrained_predictor.fit(
        train_df,
        hyperparameters={
            "env.num_gpus": args.num_gpu,
            "optimization.max_epochs": args.max_epochs,
            "model.hf_text.checkpoint_name": args.pretrained_model,
            "optimization.learning_rate": 1.0e-4,
            "optimization.weight_decay": 1.0e-3,
        },
        time_limit=args.time_limit,
        seed=args.seed,
    )
    pretrained_result = pretrained_predictor.evaluate(data=valid_df, metrics=GLUE_METRICS[glue_task]["eval"])

    ### Distill and evaluate a student model
    from autogluon.text.automm.constants import MODEL, DATA, OPTIMIZATION, ENVIRONMENT, DISTILLER
    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        DISTILLER: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }
    student_predictor = AutoMMPredictor(label="label", eval_metric=GLUE_METRICS[glue_task]["val"])
    student_predictor.fit(
        train_df,
        config=config,
        hyperparameters={
            "env.num_gpus": args.num_gpu,
            "optimization.max_epochs": args.max_epochs,
            "model.hf_text.checkpoint_name": args.pretrained_model,
            "optimization.learning_rate": 1.0e-4,
            "optimization.weight_decay": 1.0e-3,
            "distiller.temperature": args.temperature,
            "distiller.hard_label_weight": args.hard_label_weight,
            "distiller.soft_label_weight": args.soft_label_weight,
        },
        teacher_predictor=teacher_predictor,
        time_limit=args.time_limit,
        seed=args.seed,
    )
    start = time()
    student_result = student_predictor.evaluate(data=valid_df, metrics=GLUE_METRICS[glue_task]["eval"])
    student_usedtime = time() - start

    ### Print distillation's performance
    print("Distillation Result:")
    print("Teacher Model: %s" % args.teacher_model)
    print("Student Model: %s" % args.pretrained_model)
    for k in teacher_result:
        print("Teacher Model's %s: %.6f" % (k, teacher_result[k]))
        print("Pretrained Model's %s: %.6f" % (k, pretrained_result[k]))
        print("Student Model's %s: %.6f" % (k, student_result[k]))
        print(
            "Distillation Ratio (the fraction of the teacher's performance achieved by the student): %.6f"
            % (
                float(student_result[k] - pretrained_result[k])
                / float(teacher_result[k] - pretrained_result[k])
            )
        )
    print("Teacher Model's time: %.6f" % teacher_usedtime)
    print("Student Model's time: %.6f" % student_usedtime)
    print("speed up: %.6fx" % (teacher_usedtime / student_usedtime))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glue_task", default="qnli", type=str)
    parser.add_argument("--teacher_model", default="google/bert_uncased_L-12_H-768_A-12", type=str)
    parser.add_argument("--pretrained_model", default="google/bert_uncased_L-4_H-768_A-12", type=str)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--time_limit", default=7200, type=int)
    parser.add_argument("--num_gpu", default=-1, type=int)
    parser.add_argument("--temperature", default=5, type=float)
    parser.add_argument("--hard_label_weight", default=0.1, type=float)
    parser.add_argument("--soft_label_weight", default=1, type=float)
    args = parser.parse_args()

    main(args)
