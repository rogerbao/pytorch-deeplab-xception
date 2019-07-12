CUDA_VISIBLE_DEVICES=0 python train.py --backbone mobilenet --lr 0.01 --workers 4 --epochs 30 \
--batch-size 16 --checkname deeplab-mobilenet-128-step10 --eval-interval 1 --dataset pascal \
--base-size 128 --crop-size 128 --lr-scheduler step --lr-step 10

#CUDA_VISIBLE_DEVICES=0 python train.py --backbone mobilenet --lr 0.01 --workers 4 --epochs 30 \
#--batch-size 16 --checkname deeplab-mobilenet-128-test --eval-interval 1 --dataset pascal \
#--base-size 128 --crop-size 128

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --backbone resnet --lr 0.007 --workers 4 --use-sbd True --epochs 50 \
# --batch-size 16 --gpu-ids 0,1,2,3 --checkname deeplab-resnet --eval-interval 1 --dataset pascal
