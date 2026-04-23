srun --partition=ood --nodes=1 --ntasks=2 --gres=gpu:t4:2 --cpus-per-task=8 --mem=30G --time=00:30:00 --pty bash

ssh -N -L 8265:localhost:8265 $USER@
# uv add nccl4py[cu12]

source .venv/bin/activate
module load openmpi/4.1.5


mpirun -n 2 python main.py


salloc --partition=ood --nodes=1 --ntasks=2 --gres=gpu:t4:2 \
       --cpus-per-task=2 --mem=30G --time=00:30:00

srun uv run python main.py