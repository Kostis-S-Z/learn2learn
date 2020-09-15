## Meta-World

Meta-World is an open source simulated environment consisting of 50 handcrafted tasks on a Sawyer arm robot. 
This environment functions as meta-reinforcement learning and multi-task learning benchmark.

**_Note:_** The `done` signal is often used in Gym environments to indicate that either the task was completed or the
episode reached the maximum path length and thus, the environment should `reset`. In the original Meta-World environment
the functionality of the `done` signal has been disabled. This is for two reasons: 1) Even if the robot manages to solve
the task, the episode should not terminate, but most importantly, 2) calling `done` at `max_path_length` can cause
instability and confusion in off-policy algorithms. Off-policy algorithms distinguish the cases when an episode
terminates because of the task being completed and when it reaches the end of the horizon.
([more](https://github.com/rlworkgroup/metaworld/issues/84))

However, in order to make Meta-World compatible with the existing code base of learn2learn and cherry-rl, the wrapper of
Meta-World in l2l sends `done=True` when `max_path_length` is reached. Since most of the RL examples are with on-policy
algorithms this does not cause an issue. It would be something to consider in the future, if off-policy RL examples were
to be added.

_The gif was created by a trained policy of `maml_trpo` on ML1-Push-v1._