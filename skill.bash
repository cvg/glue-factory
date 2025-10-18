--------------train--------------
nohup env CUDA_VISIBLE_DEVICES=0,1 python -m gluefactory.train sp+lg_selfda_homography --distributed True --conf gluefactory/configs/superpoint+lightglue_homography.yaml > train.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=0,1 python -m gluefactory.train sp+lg_megadepth --distributed True --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml train.load_experiment=sp+lg_homography > finetune_32bs.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=0,1 python -m gluefactory.train sp+lg_megadepth --distributed True --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml --restore > train_resume.log 2>&1 &

--------------test--------------
nohup python -m gluefactory.eval.megadepth1500 --checkpoint sp+lg_da_megadepth_cross > megadepth1500.log 2>&1 &
nohup python -m gluefactory.eval.hpatches --checkpoint sp+lg_da_megadepth_cross > hpatches.log 2>&1 &
nohup python -m gluefactory.eval.scannet1500 --checkpoint sp+lg_megadepth model.matcher.{depth_confidence=0.95,width_confidence=0.95} > scannet1500.log 2>&1