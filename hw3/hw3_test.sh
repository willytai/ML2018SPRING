wget -O merge_tmp.h5 'https://www.dropbox.com/s/43sv8rfywzf3k46/merge_tmp.h5?dl=1'
wget -O merge.h5 'https://www.dropbox.com/s/wgj7wn9btczqmyg/merge.h5?dl=1'
python3 test.py $1 $3 $2
