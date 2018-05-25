import numpy as np
import re, sys

def read_test(filename):
	test   = []
	header = True
	with open(filename, 'r') as f:
		for line in f:

			# skip the header
			if header == True:
				header = False
				continue

			# find the first comma
			i = 0
			while line[i] != ',':
				i += 1
			
			# slice the string
			line = line[i+1:]

			if isEnglish(line):
				# save the sentence
				test.append(text_to_wordlist(line))
	# test = test.split(' ')

	return test


def read_train(filename, label):
	###############################
	#### training data as list ####
	####   label as 1D array   ####
	###############################

	labels = []
	train  = []
	with open(filename, 'r') as f:
		for line in f:
			
			if label == True:
				# split the label and the sentence
				line = line.split(' +++$+++ ')

			if label == True:
				labels.append(int(line[0]))
				train.append(text_to_wordlist(line[1]))
			else:
				if isEnglish(line):
					train.append(text_to_wordlist(line))
					if train[-1] == '':
						del train[-1]

	# train = train.split(' ')

	return train, np.array(labels)

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.?\/'+-=]", " ", text)
    text = re.sub(r"([^!])(!)", r'\1 \2', text)
    text = re.sub(r" too ", " tofuckyou ", text)
    text = re.sub(r" good ", " gofuckyou ", text)
    text = re.sub(r'([A-Za-z^,!,?\'+-=])\1+', r'\1', text)
    text = re.sub(r" tofuckyou ", " too ", text)
    text = re.sub(r" gofuckyou ", " good ", text)
    text = re.sub(r" \' ", "\'", text)
    text = re.sub(r"ä ±\' m", "i am", text)
    text = re.sub(r" u ", " you ", text)
    text = re.sub(r" tho ", " though ", text)
    text = re.sub(r" thru ", " through ", text)
    text = re.sub(r" fkd ", " fucked ", text)
    text = re.sub(r"try 2", "try to ", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r" ive ", " i have ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'l", " will ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\.", " .", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"\"", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # filter out weird string
    text = text.split(' ')

    text = list(filter(lambda x: x!= '', text))

    # this will happen only when reading training_nolabel.txt
    if len(text) > 100:
        text = ''
    
    return text

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def main():
    print (text_to_wordlist('i like this too,,,, god good!!!!!~')); sys.exit()
    with open('database/training_nolabel.txt') as f:
        for line in f:
            line = line.split(' +++$+++ ')[-1]
            print (line, end='')
            print (" ".join(text_to_wordlist(line)))
            print ('')

if __name__ == '__main__':
	main()
else:
	print ('Using Functions In Util')
