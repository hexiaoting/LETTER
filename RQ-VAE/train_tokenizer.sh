python ./RQ-VAE/main.py \
  --device cuda:0 \
  --data_path /home/hewenting/dataprocess/Beauty.emb-bert-base-uncased-td.npy \
  # --alpha 0.01 \
  # --beta 0.0001 \
  --ckpt_dir ../checkpoint/

  # --cf_emb ./RQ-VAE/ckpt/Instruments-32d-sasrec.pt\