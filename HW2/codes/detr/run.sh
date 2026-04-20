CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --coco_path /mnt/HDD4/anlt/data/nycu-hw2-data \
  --dataset_file coco \
  --batch_size 32 \
  --epochs 150 \
  --backbone resnet50 \
  --output_dir outputs/detr_resnet50