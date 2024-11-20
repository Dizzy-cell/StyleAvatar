CUDA_VISIBLE_DEVICES=0 python test_sup.py --input_dir ../datasets/id2/pred --ckpt *.pt --save_dir output/superresolution --mode 1

CUDA_VISIBLE_DEVICES=0 python train_sup.py --batch 3 --mode 1 --augment -augment_p 0.01 ../dataset/id2 --ckpt *.pt
