#!/usr/bin/env bash
set -euo pipefail


REPO_URL="https://github.com/reuvenperetz/LLDB.git"
DOCKER_IMAGE="577004484676.dkr.ecr.us-east-1.amazonaws.com/volt-dev:lldbv2"
NUM_GPUS="1"
GPU_TYPE="H100"
JOB_NAME="${JOB_NAME:-lldb-$(date +%m%d-%H%M)}"


TMP_SCRIPT=$(mktemp)
cat > "$TMP_SCRIPT" <<'JOB'
#!/usr/bin/env bash
set -euo pipefail

DEBUG_MODE=0


cd /tmp
mkdir ntire-llie
mkdir /tmp/ntire-llie/train
mkdir /tmp/ntire-llie/val

mkdir /tmp/ntire-llie/train/low
mkdir /tmp/ntire-llie/val/high
mkdir /tmp/ntire-llie/val/low

# Download dataset
aws s3 cp s3://$S3_BUCKET/ntire-llie/Denoised_LLIE_val_gt.zip .
aws s3 cp s3://$S3_BUCKET/ntire-llie/Denoised_LLIE_val_in.zip .

# Unzip datasets
unzip Denoised_LLIE_val_gt.zip -d /tmp/ntire-llie/val/high
unzip Denoised_LLIE_val_in.zip -d /tmp/ntire-llie/val/low


if [ "$DEBUG_MODE" -eq 0 ]; then
    echo "Downloading full training dataset"

    aws s3 cp s3://$S3_BUCKET/ntire-llie/gtPatchDLL.zip .
    aws s3 cp s3://$S3_BUCKET/ntire-llie/train_low/part_aa .
    aws s3 cp s3://$S3_BUCKET/ntire-llie/train_low/part_ab .
    aws s3 cp s3://$S3_BUCKET/ntire-llie/train_low/part_ac .
    aws s3 cp s3://$S3_BUCKET/ntire-llie/train_low/part_ad .
    aws s3 cp s3://$S3_BUCKET/ntire-llie/train_low/part_ae .

    unzip -q gtPatchDLL.zip -d /tmp/ntire-llie/train/
    mv /tmp/ntire-llie/train/gtPatchDLL /tmp/ntire-llie/train/high

    cat part_* > inPatchDLL.zip
    unzip -q inPatchDLL.zip -d /tmp/ntire-llie/train/

    rm part_aa part_ab part_ac part_ad part_ae gtPatchDLL.zip inPatchDLL.zip

else
    echo "DEBUG mode: using validation data as training data"

    # Reuse val data for training
    cp -r /tmp/ntire-llie/val/high /tmp/ntire-llie/train/high
    cp -r /tmp/ntire-llie/val/low  /tmp/ntire-llie/train
fi

export MLFLOW_TRACKING_URI="http://arm-aair-reuper01-mlflow-svc.default.svc.cluster.local:5000"
export MLFLOW_EXPERIMENT="lldb"
export JOB_NAME="job_name"

cd /dev/shm/LLDB/
git checkout ntire

export PYTHONPATH="/dev/shm/LLDB"
python /dev/shm/LLDB/codes/tasks/lol-v1/train.py -opt=/dev/shm/LLDB/codes/tasks/lol-v1/options/train.yml



JOB

GPU_TYPE_ARG=""
if [[ -n "$GPU_TYPE" ]]; then
  GPU_TYPE_ARG="--gpu_type $GPU_TYPE"
fi

run_job \
  --job_name "$JOB_NAME" \
  --local_script "$TMP_SCRIPT" \
  --repo_url "$REPO_URL" \
  --docker_image "$DOCKER_IMAGE" \
  --number_of_gpus "$NUM_GPUS" \
  --upload_outputs lldb_experiments/"$JOB_NAME" \
  $GPU_TYPE_ARG

rm -f "$TMP_SCRIPT"
