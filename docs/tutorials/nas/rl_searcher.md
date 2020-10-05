# Demo RL Searcher 
:label:`sec_rlsearcher`

In this tutorial, we are going to compare RL searcher with random search in a simulation environment.

## A Toy Reward Space

Input Space `x = [0: 99], y = [0: 99]`.
The rewards are a combination of 2 gaussians as shown in the following figure:

```{.python .input}
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```

Generate the simulation rewards as a mixture of 2 gaussians:

```{.python .input}
def gaussian(x, y, x0, y0, xalpha, yalpha, A): 
    return A * np.exp( -((x-x0)/xalpha)**2 -((y-y0)/yalpha)**2) 

x, y = np.linspace(0, 99, 100), np.linspace(0, 99, 100) 
X, Y = np.meshgrid(x, y)

Z = np.zeros(X.shape) 
ps = [(20, 70, 35, 40, 1),
      (80, 40, 20, 20, 0.7)]
for p in ps:
    Z += gaussian(X, Y, *p)
```

Visualize the reward space:

```{.python .input}
fig = plt.figure()
ax = fig.gca(projection='3d') 
ax.plot_surface(X, Y, Z, cmap='plasma') 
ax.set_zlim(0,np.max(Z)+2)
plt.show()
```

## Simulation Experiment

### Customize Train Function

We can define any function with a decorator `@ag.args`, which converts the function to
AutoGluon searchable. The `reporter` is used to communicate with AutoGluon search algorithms.

```{.python .input}
import autogluon.core as ag

@ag.args(
    x=ag.space.Categorical(*list(range(100))),
    y=ag.space.Categorical(*list(range(100))),
)
def rl_simulation(args, reporter):
    x, y = args.x, args.y
    reporter(accuracy=Z[y][x])
```

### Random Search Baseline

```{.python .input}
random_scheduler = ag.scheduler.FIFOScheduler(rl_simulation,
                                              resource={'num_cpus': 1, 'num_gpus': 0},
                                              num_trials=300,
                                              reward_attr='accuracy')
random_scheduler.run()
random_scheduler.join_jobs()
print('Best config: {}, best reward: {}'.format(random_scheduler.get_best_config(), random_scheduler.get_best_reward()))
```

### Reinforcement Learning

```{.python .input}
rl_scheduler = ag.scheduler.RLScheduler(rl_simulation,
                                        resource={'num_cpus': 1, 'num_gpus': 0},
                                        num_trials=300,
                                        reward_attr='accuracy',
                                        controller_batch_size=4,
                                        controller_lr=5e-3,
                                        checkpoint='./rl_exp/checkerpoint.ag')
rl_scheduler.run()
rl_scheduler.join_jobs()
print('Best config: {}, best reward: {}'.format(rl_scheduler.get_best_config(), rl_scheduler.get_best_reward()))
```

### Compare the Performance

Get the result history:

```{.python .input}
results_rl = [v[0]['accuracy'] for v in rl_scheduler.training_history.values()]
results_random = [v[0]['accuracy'] for v in random_scheduler.training_history.values()]
```

Average result every 10 trials:

```{.python .input}
import statistics
results1 = [statistics.mean(results_random[i:i+10]) for i in range(0, len(results_random), 10)]
results2 = [statistics.mean(results_rl[i:i+10]) for i in range(0, len(results_rl), 10)]
```

Plot the results:

```{.python .input}
plt.plot(range(len(results1)), results1, range(len(results2)), results2)
```
