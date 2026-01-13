import argparse
import os
from time import time

import pandas as pd
from datasets import load_dataset

from autogluon.multimodal import MultiModalPredictor

PAWS_TASKS = ["en", "de", "es", "fr", "ja", "ko", "zh"]


def tasks_to_id(pawsx_tasks):
    id = ""
    for task in PAWS_TASKS:
        if task in pawsx_tasks:
            id += task
    return id


def getDatasetSplits(pawsx_tasks):
    datasets = {}
    train_dfs = {}
    val_dfs = {}
    test_dfs = {}
    for task in pawsx_tasks:
        datasets[task] = load_dataset("paws-x", task)
        train_dfs[task] = datasets[task]["train"].to_pandas()
        val_dfs[task] = datasets[task]["validation"].to_pandas()
        test_dfs[task] = datasets[task]["test"].to_pandas()
        print(
            "task %s: train %d, val %d, test %d"
            % (task, len(train_dfs[task]), len(val_dfs[task]), len(test_dfs[task]))
        )
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_dfs["all"] = pd.concat(test_dfs)

    return train_df, val_df, test_dfs


def main(args):
    pawsx_teacher_tasks = args.pawsx_teacher_tasks
    assert all(task in PAWS_TASKS for task in pawsx_teacher_tasks)
    teacher_tasks_id = tasks_to_id(pawsx_teacher_tasks)
    pawsx_student_tasks = args.pawsx_student_tasks
    assert all(task in PAWS_TASKS for task in pawsx_student_tasks)
    student_tasks_id = tasks_to_id(pawsx_student_tasks)

    teacher_train_df, teacher_val_df, teacher_test_dfs = getDatasetSplits(pawsx_teacher_tasks)
    student_train_df, student_val_df, student_test_dfs = getDatasetSplits(pawsx_student_tasks)

    teacher_predictor_name = f"pawsx-{teacher_tasks_id}-{args.teacher_model.replace('/', '-')}"
    teacher_predictor_path = os.path.join(args.save_path, teacher_predictor_name)
    nodistill_predictor_name = f"pawsx-{student_tasks_id}-{args.student_model.replace('/', '-')}"
    nodistill_predictor_path = os.path.join(args.save_path, nodistill_predictor_name)

    teacher_result = {}
    nodistill_result = {}
    student_result = {}

    ### Train and evaluate the teacher model
    resume_teacher = args.resume
    try:
        teacher_predictor = MultiModalPredictor.load(teacher_predictor_path)
        print("Using pretrained teacher model: %s" % teacher_predictor_path)
    except:
        resume_teacher = False
        print("No pretrained model at: %s" % teacher_predictor_path)
    if not resume_teacher:
        teacher_predictor = MultiModalPredictor(label="label", eval_metric="accuracy")
        teacher_predictor.fit(
            teacher_train_df,
            tuning_data=teacher_val_df,
            hyperparameters={
                "env.num_gpus": args.num_gpu,
                "env.precision": args.precision,
                "env.per_gpu_batch_size": 6,
                "model.hf_text.checkpoint_name": args.teacher_model,
                "optim.lr": 1.0e-4,
                "optim.weight_decay": 1.0e-3,
            },
            time_limit=args.time_limit,
            seed=args.seed,
        )
        teacher_predictor.save(teacher_predictor_path)
    for test_name, test_df in teacher_test_dfs.items():
        teacher_result[test_name] = teacher_predictor.evaluate(data=test_df, metrics="accuracy")
    # use same dataset to measure computation time
    start = time()
    for test_name, test_df in student_test_dfs.items():
        teacher_predictor.evaluate(data=test_df, metrics="accuracy")
    teacher_usedtime = time() - start

    ### Train and evaluate a smaller pretrained model
    resume_nodistill = args.resume
    try:
        nodistill_predictor = MultiModalPredictor.load(nodistill_predictor_path)
        print("Using pretrained nodistill model: %s" % nodistill_predictor_path)
    except:
        print("No pretrained model at: %s" % nodistill_predictor_path)
        resume_nodistill = False
    if not resume_nodistill:
        nodistill_predictor = MultiModalPredictor(label="label", eval_metric="accuracy")
        nodistill_predictor.fit(
            student_train_df,
            tuning_data=student_val_df,
            hyperparameters={
                "env.num_gpus": args.num_gpu,
                "env.precision": args.precision,
                "optim.max_epochs": args.max_epochs,
                "model.hf_text.checkpoint_name": args.student_model,
                "optim.lr": 2.0e-4,
                "optim.weight_decay": 2.0e-3,
            },
            time_limit=args.time_limit,
            seed=args.seed,
        )
        nodistill_predictor.save(nodistill_predictor_path)
    for test_name, test_df in student_test_dfs.items():
        nodistill_result[test_name] = nodistill_predictor.evaluate(data=test_df, metrics="accuracy")

    ### Distill and evaluate a student model
    from autogluon.multimodal.constants import DATA, DISTILLER, ENV, MODEL, OPTIM

    config = {
        MODEL: f"default",
        DATA: "default",
        DISTILLER: "default",
        OPTIM: "default",
        ENV: "default",
    }
    student_predictor = MultiModalPredictor(label="label", eval_metric="accuracy")
    student_predictor.fit(
        student_train_df,
        tuning_data=student_val_df,
        config=config,
        hyperparameters={
            "env.num_gpus": args.num_gpu,
            "env.precision": args.precision,
            "optim.max_epochs": args.max_epochs,
            "model.hf_text.checkpoint_name": args.student_model,
            "model.hf_text.text_trivial_aug_maxscale": args.aug_scale,
            "optim.lr": 2.0e-4,
            "optim.weight_decay": 2.0e-3,
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
            "model.hf_text.text_trivial_aug_maxscale": args.aug_scale,
            # "optim.top_k": 1,
            # "optim.top_k_average_method": "best",
        },
        teacher_predictor=teacher_predictor,
        time_limit=args.time_limit,
        seed=args.seed,
    )
    start = time()
    for test_name, test_df in student_test_dfs.items():
        student_result[test_name] = student_predictor.evaluate(data=test_df, metrics="accuracy")
    student_usedtime = time() - start

    ### Print distillation's performance
    for test_name in student_test_dfs.keys():
        print("Distillation Result (%s):" % test_name)
        print("Teacher Model: %s" % args.teacher_model)
        print("Student Model: %s" % args.student_model)
        for k in teacher_result[test_name]:
            print(f"For metric {k}:")
            print("Teacher Model's %s: %.6f" % (k, teacher_result[test_name][k]))
            print("Pretrained Model's %s: %.6f" % (k, nodistill_result[test_name][k]))
            print("Student Model's %s: %.6f" % (k, student_result[test_name][k]))
            print(
                "Distillation Ratio (the fraction of the teacher's performance achieved by the student): %.6f"
                % (
                    float(student_result[test_name][k] - nodistill_result[test_name][k])
                    / float(teacher_result[test_name][k] - nodistill_result[test_name][k])
                )
            )
    print("Teacher Model's time: %.6f" % teacher_usedtime)
    print("Student Model's time: %.6f" % student_usedtime)
    print("speed up: %.6fx" % (teacher_usedtime / student_usedtime))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pawsx_teacher_tasks", nargs="+", default=["en", "de", "es", "fr", "ja", "ko", "zh"])
    parser.add_argument("--pawsx_student_tasks", nargs="+", default=["en", "de", "es", "fr", "ja", "ko", "zh"])
    parser.add_argument("--teacher_model", default="microsoft/mdeberta-v3-base", type=str)
    parser.add_argument("--student_model", default="nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large", type=str)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--precision", default="bf16", type=str)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--time_limit", default=None, type=int)
    parser.add_argument("--num_gpu", default=-1, type=int)
    parser.add_argument("--temperature", default=5.0, type=float)
    parser.add_argument("--hard_label_weight", default=0.1, type=float)
    parser.add_argument("--soft_label_weight", default=1.0, type=float)
    parser.add_argument("--softmax_regression_weight", default=0, type=float)
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
    parser.add_argument("--aug_scale", default=0.0, type=float)
    args = parser.parse_args()
    main(args)
