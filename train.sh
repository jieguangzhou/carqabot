carbot_data_path=carbot_data

rm -rf $carbot_data_path
echo $carbot_data_path

cp -r data/kg $carbot_data_path
cp -r data/kg $carbot_data_path
cp -r data/sample_question $carbot_data_path
cp -r data/dictionary $carbot_data_path
cp -r data/complex_qa.xlsx $carbot_data_path


python bert_script/run_text_classifier.py \
  --cache_dir ./data/bert_pretrain \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/train/kbqa/predicate_classification \
  --bert_model bert-base-chinese \
  --max_seq_length 30 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir $carbot_data_path/model/kbqapc/

python bert_script/run_sequence_labeling.py \
  --cache_dir ./data/bert_pretrain \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/train/kbqa/ner \
  --bert_model bert-base-chinese \
  --max_seq_length 30 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir $carbot_data_path/model/ner/

python bert_script/run_text_similarity.py \
  --text_a_key question \
  --text_b_key predicate \
  --label_key label \
  --cache_dir ./data/bert_pretrain \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/train/kbqa/predicate_match \
  --bert_model bert-base-chinese \
  --max_seq_length 30 \
  --train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir $carbot_data_path/model/kbqapm/

python script/get_relation_match_data.py \
  --sample_qa_path data/train/kbqa/predicate_classification/train.xlsx \
  --complex_qa_path data/complex_qa.xlsx \
  --save_path data/train/kbqa/complex_qa \


python bert_script/run_text_similarity.py \
  --text_a_key text_a \
  --text_b_key text_b \
  --label_key label \
  --cache_dir ./data/bert_pretrain \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/train/kbqa/complex_qa \
  --bert_model bert-base-chinese \
  --max_seq_length 30 \
  --train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir $carbot_data_path/model/kbqasp/


: '
python bert_script/run_squad.py \
  --bert_model bert-base-chinese \
  --cache_dir ./data/bert_pretrain \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file data/train/dbqa/train.json \
  --predict_file data/train/dbqa/test.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 128 \
  --doc_stride 128 \
  --output_dir carbot_data/model/dbqa/

 '
