echo "TRAINING SCRIPT EXECUTING"

python train.py --data corpus_all_train.csv --test_data corpus_all_test.csv --output trained_model

echo "TEST SCRIPT EXECUTING" 

python infer.py --model ./trained_model --data corpus_all_test.csv 

echo "PROGRAM COMPLETE"