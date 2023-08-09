.PHONY: stage-0

stage-0:
	CUDA_VISIBLE_DEVICES=0 pipenv run python ./main.py \
	--attention_weights_only True \
	--dataset sceneflow \
	--datapath ~/sda/sceneflow/ \
	--trainlist ~/repo/myACVNet/file_lists/sceneflow_train.txt \
	--testlist ~/repo/myACVNet/file_lists/sceneflow_test.txt \
	--logdir ~/.tensorboard/acv-origin/train/stage-0 \
	--loadckpt '' \
	--summary_freq 40 \
	--batch_size 9 \
	--test_batch_size 9 \
	--lr 0.001 \

stage-1:
	./main.py \
	--freeze_attention_weights True \
	--dataset sceneflow \
	--datapath /data/sceneflow/ \
	--trainlist filenames/sceneflow_train.txt \
	--logdir .log/stage_2 \
	--loadckpt ./log/stage_1/checkpoint_000063.ckpt \
	--batch_size 14

stage-2:
	./main.py \
	--dataset sceneflow \
	--datapath /data/sceneflow/ \
	--trainlist filenames/sceneflow_train.txt \
	--logdir .log/stage_3 \
	--loadckpt ./log/stage_2/checkpoint_000063.ckpt \
	--batch_size 10