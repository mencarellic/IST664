import nltk
import re

# Creating stopwords var and adding some extras for old english
stopWords = set(nltk.corpus.stopwords.words('english'))
stopWords.update(['thou', 'doe', 'thy', 'thee', 'hath', 'shall', 'thus', 'ye', 'vpon', 'vs', 'heere',
                  'haue', 'qu', 'exeunt', 'enter'])

# Creating bigram and trigram measures
bigramMeasures = nltk.collocations.BigramAssocMeasures()
trigramMeasures = nltk.collocations.TrigramAssocMeasures()

# Assigning raw texts to variables for Paradise Lost and Hamlet
# Also stripping the right side whitespace and dropping everything to lowercase
milton = nltk.corpus.gutenberg.raw('milton-paradise.txt').rstrip().lower()
shakespeare = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt').rstrip().lower()

# Because Hamlet is setup as a play, there's some abbreviated names for dialogue.
# Need to remove these to not skew counts
shakespeareRemoval = ['ham\.', 'hora?\.', 'mar\.', 'ber\.', 'fran\.', 'ophe?\.', 'pol\.', 'laer\.', 'barn?\.',
                      'forn?\.', 'amb\.', 'osr\.', 'fort\.', 'rey\.', 'ros\.', 'guil\.', 'king?\.', 'queen\.',
                      'qu\.', 'clo\.', 'rosin\.', 'polon\.', 'reynol\.', 'marcell\.', 'volt\.']
shakespearePattern = re.compile('|^\s+'.join(shakespeareRemoval), re.MULTILINE)
shakespeare = shakespearePattern.sub("", shakespeare)

# Tokenizing both Paradise Lost and Hamlet
print('\n-------\nTokenizing Milton\n-------')

miltonToken = re.sub('\'', '', milton)
miltonToken = nltk.word_tokenize(re.sub('\W+',' ', miltonToken))
miltonNum = len(miltonToken)
miltonSet = set(miltonToken)
print('Milton token count: {}'.format(miltonNum))
print('Milton set count: {}'.format(len(miltonSet)))
print('Milton token sample:\n{}'.format(miltonToken[:100]))

print('\n-------\nTokenizing Shakespeare\n-------')

shakespeareToken = re.sub('\'', '', shakespeare)
shakespeareToken = nltk.word_tokenize(re.sub('\W+',' ', shakespeareToken))
shakespeareNum = len(shakespeareToken)
shakespeareSet = set(shakespeareToken)
print('Shakespeare token count: {}'.format(shakespeareNum))
print('Shakespeare set count: {}'.format(len(shakespeareSet)))
print('Shakespeare token sample:\n{}'.format(shakespeareToken[:100]))

# Is it worth turning integers to words for these texts?
print('\n-------\nInteger checking\n-------')

miltonInts = [num for num in miltonToken if re.match('\d', num)]
print('Integers in Milton: {}'.format(miltonInts))

shakespeareInts = [num for num in shakespeareToken if re.match('\d', num)]
print('Integers in Shakespeare: {}'.format(shakespeareInts))


# Removing stop words from corpora
miltonStopped = [word for word in miltonToken if word not in stopWords]
shakespeareStopped = [word for word in shakespeareToken if word not in stopWords]


# Creating frequency distributions for Paradise Lost and Hamlet
print('\n-------\nTop 50 words in Milton corpus by frequency\n-------')
miltonFdist = nltk.FreqDist(miltonStopped)
miltonTop50 = miltonFdist.most_common(50)
miltonNorm = [(word,round(freq/miltonNum, 5)) for (word, freq) in miltonTop50]
for pair in miltonNorm:
    print(pair)

print('\n-------\nTop 50 words in Shakespeare corpus by frequency\n-------')
shakespeareFdist = nltk.FreqDist(shakespeareStopped)
shakespeareTop50 = shakespeareFdist.most_common(50)
shakespeareNorm = [(word,round(freq/shakespeareNum,5)) for (word,freq) in shakespeareTop50]
for pair in shakespeareNorm:
    print(pair)

# Creating bigram and trigram lists
print('\n-------\nTop bigrams Milton corpus by frequency\n-------')
miltonFinder = nltk.collocations.BigramCollocationFinder.from_words(miltonStopped)
miltonScored = miltonFinder.score_ngrams(bigramMeasures.raw_freq)
for bigram in miltonScored[:50]:
    print(bigram)

print('\n-------\nTop bigrams Milton corpus by PMI\n-------')
miltonFinder.apply_freq_filter(5)
miltonScored = miltonFinder.score_ngrams(bigramMeasures.pmi)
for bigram in miltonScored[:50]:
    print(bigram)

print('\n-------\nTop trigrams Milton corpus by frequency\n-------')
miltonFinder = nltk.collocations.TrigramCollocationFinder.from_words(miltonStopped)
miltonScored = miltonFinder.score_ngrams(trigramMeasures.raw_freq)
for trigram in miltonScored[:50]:
    print(trigram)

print('\n-------\nTop trigrams Milton corpus by PMI\n-------')
miltonFinder.apply_freq_filter(3)
miltonScored = miltonFinder.score_ngrams(trigramMeasures.pmi)
for trigram in miltonScored[:50]:
    print(trigram)


print('\n-------\nTop bigrams Shakespeare corpus by frequency\n-------')
shakespeareFinder = nltk.collocations.BigramCollocationFinder.from_words(shakespeareStopped)
shakespeareScored = shakespeareFinder.score_ngrams(bigramMeasures.raw_freq)
for bigram in shakespeareScored[:50]:
    print(bigram)

print('\n-------\nTop bigrams Shakespeare corpus by PMI\n-------')
shakespeareFinder.apply_freq_filter(5)
shakespeareScored = shakespeareFinder.score_ngrams(bigramMeasures.pmi)
for bigram in shakespeareScored[:50]:
    print(bigram)

print('\n-------\nTop trigrams Shakespeare corpus by frequency\n-------')
shakespeareFinder = nltk.collocations.TrigramCollocationFinder.from_words(shakespeareStopped)
shakespeareScored = shakespeareFinder.score_ngrams(trigramMeasures.raw_freq)
for trigram in shakespeareScored[:50]:
    print(trigram)

print('\n-------\nTop trigrams Shakespeare corpus by PMI\n-------')
shakespeareFinder.apply_freq_filter(3)
shakespeareScored = shakespeareFinder.score_ngrams(trigramMeasures.pmi)
for trigram in shakespeareScored[:50]:
    print(trigram)