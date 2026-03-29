#!/bin/bash
while true; do
    TRAIN_COUNT=$(find /workspace/EMA-pTTQ/data/ImageNet/train/ -name "*.JPEG" 2>/dev/null | wc -l)
    VAL_COUNT=$(find /workspace/EMA-pTTQ/data/ImageNet/val/ -name "*.JPEG" 2>/dev/null | wc -l)
    echo "$(date): train=$TRAIN_COUNT/1281167 val=$VAL_COUNT/50000" >> /workspace/EMA-pTTQ/download_status.log
    if [ "$TRAIN_COUNT" -ge 1281167 ] && [ "$VAL_COUNT" -ge 50000 ]; then
        echo "$(date): DOWNLOAD COMPLETE!" >> /workspace/EMA-pTTQ/download_status.log
        break
    fi
    # Check if process is still alive
    if ! pgrep -f "imagenet_data.py" > /dev/null 2>&1; then
        echo "$(date): PROCESS DIED! train=$TRAIN_COUNT val=$VAL_COUNT" >> /workspace/EMA-pTTQ/download_status.log
        break
    fi
    sleep 1800
done
