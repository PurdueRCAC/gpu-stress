gpu-stress
==========


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
&nbsp;
[![Version](https://img.shields.io/github/v/release/purduercac/gpu-stress?sort=semver)](https://github.com/purduercac/gpu-stress)
&nbsp;
[![Python Version](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads)


A simple command-line program to stress Nvidia GPU devices.
We use [PyTorch](https://pytorch.org) to multiply two large matrices repeatedly for some period of time.


Install
-------

The well-known [uv](https://docs.astral.sh/uv/) utility is the best way to install this program.

```sh
uv tool install gpu-stress
```

Usage
-----

Use `gpu-stress --help` to get usage and help on options.
Use something like the following to stress the GPU for 1 minute.

```sh
gpu-stress -t 1m
```