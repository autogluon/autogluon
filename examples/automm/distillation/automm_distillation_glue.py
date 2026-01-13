import argparse
import os
from time import time

from datasets import load_dataset

from autogluon.multimodal import MultiModalPredictor

GLUE_METRICS = {
    "mnli": {"val": "accuracy", "eval": ["accuracy"]},
    "qqp": {"val": "accuracy", "eval": ["accuracy", "f1"]},
    "qnli": {"val": "accuracy", "eval": ["accuracy"]},
    "sst2": {"val": "accuracy", "eval": ["accuracy"]},
    "stsb": {
        "val": "pearsonr",
        "eval": ["pearsonr", "spearmanr"],
    },  # Current default soft label loss func is for classification, should automatically select loss_func
    "mrpc": {"val": "accuracy", "eval": ["accuracy"]},
    "rte": {"val": "accuracy", "eval": ["accuracy"]},
    # "cola": "", #phi coefficient is not implemented
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glue_task", default="qnli", type=str)
    parser.add_argument("--teacher_model", default="google/bert_uncased_L-12_H-768_A-12", type=str)
    parser.add_argument("--student_model", default="google/bert_uncased_L-6_H-768_A-12", type=str)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--time_limit", default=None, type=int)
    parser.add_argument("--num_gpu", default=-1, type=int)
    parser.add_argument("--temperature", default=5.0, type=float)
    parser.add_argument("--hard_label_weight", default=0.1, type=float)
    parser.add_argument("--soft_label_weight", default=1.0, type=float)
    parser.add_argument(
        "--train_nodistill", default=True, type=bool, help="Whether to train the student model without distillation."
    )
    parser.add_argument("--softmax_regression_weight", default=0.1, type=float)
    parser.add_argument("--output_feature_loss_weight", default=0.01, type=float)
    parser.add_argument("--rkd_distance_loss_weight", default=0.0, type=float)
    parser.add_argument("--rkd_angle_loss_weight", default=0.0, type=float)
    parser.add_argument("--soft_label_loss_type", default="", type=str)
    parser.add_argument("--softmax_regression_loss_type", default="mse", type=str)
    parser.add_argument("--output_feature_loss_type", default="mse", type=str)
    parser.add_argument(
        "--save_path",
        default="./AutogluonModels/cache_finetuned",
        type=str,
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(args):
    assert args.glue_task in (list(GLUE_METRICS.keys()) + ["mnlim", "mnlimm"]), "Unsupported dataset name."

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

    teacher_predictor_name = f"{args.glue_task}-{args.teacher_model.replace('/', '-')}"
    teacher_predictor_path = os.path.join(args.save_path, teacher_predictor_name)
    nodistill_predictor_name = f"{args.glue_task}-{args.student_model.replace('/', '-')}"
    nodistill_predictor_path = os.path.join(args.save_path, nodistill_predictor_name)

    ### Train and evaluate the teacher model
    resume_teacher = args.resume
    try:
        teacher_predictor = MultiModalPredictor.load(teacher_predictor_path)
        print("Using pretrained teacher model: %s" % teacher_predictor_path)
    except:
        resume_teacher = False
        print("No pretrained model at: %s" % teacher_predictor_path)
    if not resume_teacher:
        teacher_predictor = MultiModalPredictor(label="label", eval_metric=GLUE_METRICS[glue_task]["val"])
        teacher_predictor.fit(
            train_df,
            hyperparameters={
                "env.num_gpus": args.num_gpu,
                "model.hf_text.checkpoint_name": args.teacher_model,
                "optim.lr": 1.0e-4,
                "optim.weight_decay": 1.0e-3,
            },
            time_limit=args.time_limit,
            seed=args.seed,
        )
        teacher_predictor.save(teacher_predictor_path)
    start = time()
    teacher_result = teacher_predictor.evaluate(data=valid_df, metrics=GLUE_METRICS[glue_task]["eval"])
    teacher_usedtime = time() - start

    ### Train and evaluate a smaller pretrained model
    resume_nodistill = args.resume
    try:
        nodistill_predictor = MultiModalPredictor.load(nodistill_predictor_path)
        print("Using pretrained nodistill model: %s" % nodistill_predictor_path)
    except:
        print("No pretrained model at: %s" % nodistill_predictor_path)
        resume_nodistill = False
    if not resume_nodistill and args.train_nodistill:
        nodistill_predictor = MultiModalPredictor(label="label", eval_metric=GLUE_METRICS[glue_task]["val"])
        nodistill_predictor.fit(
            train_df,
            hyperparameters={
                "env.num_gpus": args.num_gpu,
                "optim.max_epochs": args.max_epochs,
                "model.hf_text.checkpoint_name": args.student_model,
                "optim.lr": 1.0e-4,
                "optim.weight_decay": 1.0e-3,
            },
            time_limit=args.time_limit,
            seed=args.seed,
        )
        nodistill_predictor.save(nodistill_predictor_path)
        nodistill_result = nodistill_predictor.evaluate(data=valid_df, metrics=GLUE_METRICS[glue_task]["eval"])

    ### Distill and evaluate a student model

    student_predictor = MultiModalPredictor(label="label", eval_metric=GLUE_METRICS[glue_task]["val"])
    student_predictor.fit(
        train_df,
        hyperparameters={
            "env.num_gpus": args.num_gpu,
            "optim.max_epochs": args.max_epochs,
            "model.hf_text.checkpoint_name": args.student_model,
            "optim.lr": 1.0e-4,
            "optim.weight_decay": 1.0e-3,
            "distiller.temperature": args.temperature,
            "distiller.hard_label_weight": args.hard_label_weight,
            "distiller.soft_label_weight": args.soft_label_weight,
            "distiller.softmax_regression_weight": args.softmax_regression_weight,
            "distiller.output_feature_loss_weight": args.output_feature_loss_weight,
            "distiller.rkd_distance_loss_weight": args.rkd_distance_loss_weight,
            "distiller.rkd_angle_loss_weight": args.rkd_angle_loss_weight,
            "distiller.soft_label_loss_type": args.soft_label_loss_type,
            "distiller.softmax_regression_loss_type": args.softmax_regression_loss_type,
            "distiller.output_feature_loss_type": args.output_feature_loss_type,
            "model.hf_text.text_trivial_aug_maxscale": 0.0,
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
    print("Student Model: %s" % args.student_model)
    for k in teacher_result:
        print(f"For metric {k}:")
        print("Teacher Model's %s: %.6f" % (k, teacher_result[k]))
        print("Pretrained Model's %s: %.6f" % (k, nodistill_result[k]))
        print("Student Model's %s: %.6f" % (k, student_result[k]))
        print(
            "Distillation Ratio (the fraction of the teacher's performance achieved by the student): %.6f"
            % (float(student_result[k] - nodistill_result[k]) / float(teacher_result[k] - nodistill_result[k]))
        )
    print("Teacher Model's time: %.6f" % teacher_usedtime)
    print("Student Model's time: %.6f" % student_usedtime)
    print("speed up: %.6fx" % (teacher_usedtime / student_usedtime))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
