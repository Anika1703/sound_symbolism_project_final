# sound_symbolism_project_final

Data files: 
intersect: set of all IPA segments that intersects wikipron and our custom data
prep_bert_data.py: parses and clean wikipron scrape and custom data to determine IPA symbol vocabulary
corpus_clean_test.csv: cleaned test split
corpus_clean_train.csv: clead train split

Original files: 
run_dl.sh: Shells script to run and train adversarial model
train_adv.py: trains the original adversarial model
infer_adv.py: inferes the original adversarial model
train_lang_classifier.py: isolates the language classifier 
train_ss_classifier.py: isolates the sound symbolism/size classifier

Updated files that include BERT embedding: 
run_dl_bert.sh: Shell script to run and train updated adversarial model 
train_bert.py: trains and saves the BERT embedding model
train_adv_bert.py: trains the adversarial model with bert
infer_adv_bert.py: runs inference on the finished model

Zipped files: 
trained_model.zip: Trained BERT model -- should work out of the box
wikipron_combined.tsv.zip: filtered and cleaned scrapes from wikipron 