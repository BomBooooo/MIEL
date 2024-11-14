export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_id_name=traffic
model_name=MILE
seq_len=96
pred_len=96
data=custom_new


python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --label_len 96 \
  --pred_len $pred_len \
  --window_len 96 192 384 \
  --train_epochs 50 \
  --individual \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1
  #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len''$window_len.log