dataset_version=$1
dataset_directory='../PKUMMDv'$dataset_version
echo 'dataset version: '$dataset_directory

processed_directory=$dataset_directory'/Data/skeleton_processed'
echo 'preprocessed data directory: '$processed_directory
if [ -d $processed_directory ];then
echo 'data directory already exists, now remove it...'
rm -rf $processed_directory
fi

sequence_length=$2
echo 'preprocessing dataset with sequence length: '$sequence_length
python preprocess_skeleton.py $dataset_directory $sequence_length
python split_dataset.py
