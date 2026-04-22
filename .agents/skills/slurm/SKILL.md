---
name: gpu
description: slurm and ray
---

# Interactive Development

## When to Use

Usually we will use `srun` for interactive development. To pull up docs `srun -h`. 

## SRUN Examples

```bash
# for minimal weight transfer test
srun --gres=gpu:h100:2 --cpus-per-task=8 --mem=80G --time=00:15:00 --pty bash

# test ray cluster
srun --nodes=1 --cpus-per-task=8 --mem=30G --time=00:10:00 --pty bash
```

## Ray Examples

```bash
# head node
uv run ray start --head --include-dashboard yes --temp-dir $(pwd)/ray_tmp

# add nodes
uv run ray start --address='IP:6379'

# CPU nodes have alias like c123 while GPU nodes have alias like g123
This is run on the login node
ssh -N -L 8265:localhost:8265 $USER@c123
```