import os
import torch
import argparse

from pipeline import semantic_annotation_pipeline
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import BlipProcessor, BlipForConditionalGeneration

import torch.distributed as dist
import torch.multiprocessing as mp

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--data_dir', default='./images/dogs.jpg', help='specify the root path of images and masks')
    parser.add_argument('--checkpoint', default='./weights/FastSAM.pt', help='the path of fastsam checkpoint')
    parser.add_argument('--out_dir', default='./output', help='the dir to save semantic annotations')
    parser.add_argument('--save_img', default=True, action='store_true', help='whether to save annotated images')
    parser.add_argument('--world_size', type=int, default=1, help='number of nodes')
    args = parser.parse_args()
    return args

def main(rank, args):
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
  
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")#.to(rank)

    oneformer_ade20k_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    oneformer_ade20k_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")#.to(rank)
    oneformer_coco_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
    oneformer_coco_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")#.to(rank)

    # blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    # blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(rank)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")#.to(rank)
    
    clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")#.to(rank)
    clipseg_processor.image_processor.do_resize = False
   
    with torch.no_grad():
        semantic_annotation_pipeline(args.checkpoint, args.data_dir, args.out_dir, rank, save_img=args.save_img,
                                    clip_processor=clip_processor, clip_model=clip_model,
                                    oneformer_ade20k_processor=oneformer_ade20k_processor, oneformer_ade20k_model=oneformer_ade20k_model,
                                    oneformer_coco_processor=oneformer_coco_processor, oneformer_coco_model=oneformer_coco_model,
                                    blip_processor=blip_processor, blip_model=blip_model,
                                    clipseg_processor=clipseg_processor, clipseg_model=clipseg_model)
        # torch.cuda.empty_cache()
if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args.world_size > 1:
        mp.spawn(main,args=(args,),nprocs=args.world_size,join=True)
    else:
        main(0, args)