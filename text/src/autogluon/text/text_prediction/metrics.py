__all__ = ['calculate_metric_by_expr', 'infer_eval_log_metrics']

import ast
import operator as op
from autogluon.core.constants import MULTICLASS, BINARY, REGRESSION

# supported operators
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}


def infer_eval_log_metrics(problem_type, eval_metric=None):
    """Decide default evaluation, stopping, and logging metrics (based on type of prediction problem).

    Parameters
    ----------
    problem_type
        Type of the problem. Either regression, multiclass, or binary
    eval_metric
        The eval metric provided by the user

    Returns
    -------
    eval_metric
        The updated evaluation metric
    log_metrics
        The updated logging metric
    """
    if problem_type == MULTICLASS:
        if eval_metric is None:
            eval_metric = 'acc'
        log_metrics = ['acc', 'log_loss']
    elif problem_type == BINARY:
        if eval_metric is None:
            eval_metric = 'acc'
        log_metrics = ['f1', 'mcc', 'roc_auc', 'acc', 'log_loss']
    elif problem_type == REGRESSION:
        if eval_metric is None:
            eval_metric = 'rmse'
        log_metrics = ['r2', 'rmse', 'mae']
    else:
        raise NotImplementedError('The problem type is not supported yet!')
    if eval_metric not in log_metrics:
        log_metrics.append(eval_metric)
    return eval_metric, log_metrics


def eval_math_expr(expr):
    """Evaluate an expression

    Parameters
    ----------
    expr
        expression

    Returns
    -------
    ret
        Returned value

    Examples
    --------
    >>> eval_math_expr('2^6')
    4
    >>> eval_math_expr('2**6')
    64
    >>> eval_math_expr('1 + 2*3**(4^5) / (6 + -7)')
    -5.0
    """
    return eval_(ast.parse(expr, mode='eval').body)


def eval_(node):
    if isinstance(node, ast.Num):  # <number>
        return node.n
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return operators[type(node.op)](eval_(node.operand))
    else:
        raise TypeError(node)


def calculate_metric_by_expr(label_metric_scores: dict, label_names: list, expr: str) -> float:
    """Calculate the metric scores based on the given expression.

    Parameters
    ----------
    label_metric_scores
        All metric scores reported in the validation phase.
        It will be a dict of metric scores.
    label_names
        Name of the labels
    expr
        The expression. Supports different possibilities:
        - A single metric like 'acc', 'f1', or 'auc'
            This means to use this value as the final result.
            If there are multiple labels, we use the average of all the individual metrics
        - Combined metric, we use the syntax `label.metric_name` to describe the metric of
          a given label
            - expr = (acc + f1) / 2
                The average of the accuracy and f1 of all labels
            - expr = (label1.auc + label2.auc) / 2
                The average of the auc of "label1" and the auc of "label2"
            - expr = 0.8 * intent.acc + 0.2 * slot.f1
                0.8 * the accuracy of a label named "intent" +
                0.2 * the f1 score of a label named "slot"
            - expr = 0.1 * label1.f1 + 0.9 * acc
                0.1 * the F1 of label 1 + 0.9 * the average accuracy

    Returns
    -------
    score
        The returned score.
    """
    original_expr = expr
    possible_metric_names = set()
    for label_name in label_names:
        assert label_name in label_metric_scores,\
            'Invalid label_metric_scores,' \
            ' all provided labels should be in the aggregated label metric scores. ' \
            'label_names={}, label_metric_scores={}'.format(label_names, label_metric_scores)
        metric_scores = label_metric_scores[label_name]
        for metric_name, value, in metric_scores.items():
            expr = expr.replace('{}.{}'.format(label_name, metric_name), str(value))
            possible_metric_names.add(metric_name)
    for metric_name in possible_metric_names:
        if metric_name in expr:
            avg_metric = 0
            for label_name in label_names:
                avg_metric += label_metric_scores[label_name][metric_name]
            avg_metric /= len(label_names)
            expr = expr.replace(metric_name, str(avg_metric))
    try:
        ret = eval_math_expr(expr)
    except Exception:
        raise ValueError('Cannot successfully parse the given expression. '
                         'The original expression = "{}". After the parsing, it becomes {} but '
                         'still cannot be evalauted.'.format(original_expr, expr))
    return ret
