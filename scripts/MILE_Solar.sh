export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_id_name=Solar
model_name=MILE
data=custom_new


python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Solar-Energy/ \
  --data_path solar_AL.csv \
  --model_id $model_id_name'_'96'_'96 \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --window_len 96 192 384 \
  --train_epochs 50 \
  --individual \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Solar-Energy/ \
  --data_path solar_AL.csv \
  --model_id $model_id_name'_'96'_'192 \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --window_len 96 192 384 \
  --train_epochs 50 \
  --individual \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Solar-Energy/ \
  --data_path solar_AL.csv \
  --model_id $model_id_name'_'96'_'336 \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --window_len 96 192 384 \
  --train_epochs 50 \
  --individual \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Solar-Energy/ \
  --data_path solar_AL.csv \
  --model_id $model_id_name'_'96'_'720 \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --window_len 96 192 384 \
  --train_epochs 50 \
  --individual \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Solar-Energy/ \
  --data_path solar_AL.csv \
  --model_id $model_id_name'_'96'_'960 \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --window_len 96 192 384 \
  --train_epochs 50 \
  --individual \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Solar-Energy/ \
  --data_path solar_AL.csv \
  --model_id $model_id_name'_'96'_'1080 \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 1080 \
  --window_len 96 192 384 \
  --train_epochs 50 \
  --individual \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Solar-Energy/ \
  --data_path solar_AL.csv \
  --model_id $model_id_name'_'96'_'1200 \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 1200 \
  --window_len 96 192 384 \
  --train_epochs 50 \
  --individual \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Solar-Energy/ \
  --data_path solar_AL.csv \
  --model_id $model_id_name'_'96'_'1440 \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 1080 \
  --window_len 96 192 384 \
  --train_epochs 50 \
  --individual \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Solar-Energy/ \
  --data_path solar_AL.csv \
  --model_id $model_id_name'_'96'_'1560 \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 1560 \
  --window_len 96 192 384 \
  --train_epochs 50 \
  --individual \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Solar-Energy/ \
  --data_path solar_AL.csv \
  --model_id $model_id_name'_'96'_'1680 \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 1680 \
  --window_len 96 192 384 \
  --train_epochs 50 \
  --individual \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1
  #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len''$window_len.log