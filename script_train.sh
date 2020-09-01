export PYTHONUNBUFFERED=true
LOG="output/c3d/thumos14/train_log_${i}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")

python trainval_net.py
