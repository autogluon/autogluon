"""
Define a feature generator that iteratively prunes features based on CatBoost feature importance.
Firstly, compute marginal feature importance for all features.
1. Easy Case: Removing features one by one
- Remove the feature with lowest importance and repeat
- Pros: Most Accurate / Cons: Need to fit |features| models
2. Hard Case: Removing multiple features
- Remove up to X% of the features and repeat
- Pros: Need to fit |features|*X/100 models / Cons: Removing best feature subset means performing combinatorial search




Feature Importance Beam Search: Every beam search iteration, use bandit algorithm for computation allocation

def generate_candidate():
    beams = [(all features, no feature)]
    while resource remains
        for configuration in beams:
            kept features, pruned features = configuration info
            for feature in kept features:
                removal candidate = pruned features UNION feature
                all removal candidates = all removal candidates UNION removal candidate
        all removal candidate info = []
        # PORTION BELOW SHOULD BE REPLACED WITH BANDIT ALGORITHM (infinite arm bandit with reward = relevance uncertainty?)
        for removal candidate in all removal candidates
            fi = feature importance for removal candidate
            all removal candidate info = all removal candidate info UNION (removal candidate, fi)
        all removal candidate info = sorted(all removal candidate info, fi, descending)
        beams = select num beams best removal candidates from all removal candidate info


Notes
1. When fitting the model the first time on all data, keep track how long it took to fit the model.
2. Consider setting generate_candidate's resource to Time(model fit) and maximum prune percentage to 5%.
3. 

Rationale

If trying naive LOCO without feature importance, removing one feature takes |features| * Time(model fit).
If we try an approach where we repeatedly perform FI computation until one of the features are sufficiently
deemed irrelevant (ex. p-value) and try model fit, it takes Time(feature importance) * |features| + Time(model fit).
As long as Time(feature importance) * |features| << (|features|-1) * Time(model fit), we have computation saving.
Time(feature importance) denotes time it takes to compute feature importance for a single feature.
However, refitting everytime you remove a feature is infeasible for high dimensional datasets (ex |features| > 50)
since there may be many useless features. A more scalable solution is being able to remove multiple features
at once using FI metrics. Let's assume we want to be able to remove up to 5% of features at once. Naive but
optimal removal + refit will take O(2^Time(model fit)) time, which is unacceptable. Naive but optimal FI-based
removal will take O(2^Time(feature importance)) + Time(model) time, which is also unacceptable. Backward search
removal + refit takes Time(model fit) * 0.05 |features|^2 time while backward search pruning using FI metrics
takes Time(model fit) + Time(feature importance) * 0.05 |features|^2. Backward search based methods do not return
the best feature configuration since they are greedy but account for feature interaction. Super greedy FI based pruning takes
Time(feature importance) * |features| + Time(model fit) time. However, this does not take into account interactions
between features (ex. if pruned feature 1 is only CI with label given pruned feature 2) and is prone to overpruning.


Question: Can A be pruned (ie A |_ T | B,C is true) and suddenly B /|_ T | C if previously B |_ T | A,C?
Scenario:
- Pruned Features: []
- Kept Features: [A, B, C]
- Target: [T]

A being pruned means A is unobserved. The question here is equivalent to if A being unobserved
creates a path in d-separation sense. This scenario can happen if A was a head-to-tail or
tail-to-tail node and is not direcly connected to T. Or, A was blocking a path from T to B.
But by definition, if A was conditionally independent from T, there was no path from T to A
given other variables (and maybe B). However, since B was conditionally independent from T given
other variables as well (it's not A because A there is no path from T to A), removing A should have no effect
on CI status of B and T. Essentially proof by contradiction: for the question to be true the two
variables must be mutually blocking the only available d-separation path to target variable from each other.
But that's not possible.

Conclusion: Pruning a node will not cause a currently kept but useless node to suddenly be dependent
on the target variable. Summary: A |_ T | B,C AND B |_ T | A,C => B |_ T | C

Question: Can B be pruned (ie B |_ T | C) and suddenly A /|_ T | C if previously A |_ T | B, C?
Scenario:
- Pruned Features: [A]
- Kept Features: [B, C]
- Target: [T]

"""

"""
TODO
1. Make custom feature generator pipeline
2. Call normal fit on autogluon tabular predictor
3. Make sure things run on adult/australian

TODO
1. Get n_repeats to work
2. Figure out why feature importance computation take SO LONG DAMN IT
3. Run experiments on fit_with_prune stuff again, but get backwardsearch to actually work by reducing subsample_size to 1000?
"""

import argparse
import pandas as pd
from autogluon.features.generators import PipelineFeatureGenerator, ProxyModelFeatureSelector
from autogluon.tabular import TabularPredictor


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--train_path', help='train dataset path', type=str, default='dataset/australian/train_data.csv')
parser.add_argument('-g', '--test_path', help='train dataset path', type=str, default='dataset/australian/test_data.csv')
parser.add_argument('-l', '--label', help='name of the label column', type=str, default='class')
parser.add_argument('-p', '--prune', help='to use fit_with_prune or not', dest='prune', action='store_true')
parser.add_argument('-s', '--stack', help='use multi-layer stacking', dest='stack', action='store_true')
args = parser.parse_args()

train_data = pd.read_csv(args.train_path).head(50000)
test_data = pd.read_csv(args.test_path)
y_test = test_data[args.label]

# custom_generator = ProxyModelFeatureSelector(model="CatBoost")
# feature_generator = PipelineFeatureGenerator(generators=[[custom_generator]])
fit_args = {
    'verbosity': 2,
    'presets': ['best_quality'] if args.stack else ['medium_quality_faster_train'],
    'time_limit': 600,
    'num_bag_sets': 1 if args.stack else None,
}
if args.prune:
    fit_args['_feature_generator_kwargs'] = {'enable_feature_selection': True}

predictor = TabularPredictor(label=args.label)
try:
    predictor = predictor.fit(train_data, **fit_args)
except:
    import pdb; pdb.post_mortem()
y_pred = predictor.predict(test_data)
performance = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
print(performance)
