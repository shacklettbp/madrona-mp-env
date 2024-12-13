MadronaMPEnv is an experimental learning environment for training AI agents to play competitive multiplayer games with reinforcement learning. This environment is part of an ongoing research collaboration between Activision and the Graphics Lab at Stanford University. Currently, the codebase is an active work in progress; we plan to provide plug-and-play setup instructions in a future update.

This environment is built on top of the [Madrona Engine](https://madrona-engine.github.io), a prototype game engine for building high-performance learning environments that can execute at millions of frames per second by parallelizing gameplay logic on the GPU.

The training code in this repo (scripts/jax\_train.py) depends on [JAX](https://github.com/jax-ml/jax), a deep-learning framework that allows us to train multiple neural network based agents simultaneously. These agents compete with each other, eventually developing emergent skills and strategies that lead to success.
