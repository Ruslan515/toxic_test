from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score
import re

def regexp_tokenizer(texts, lemmatizer, stop_words):
    result = []
    with tqdm(total=len(texts)) as progress_bar:
        for t in texts:
            lemmatized_words = []
            text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                      'url', t)
            text = re.sub('_', ' ', text)
            text = re.sub('\d+', '', text)
            tokens = re.findall('''\w+\d+\w+?|\d+\w+|(?:\w\s)+|(?:\w\s?\.\s?)+\w?|\w+\.+\w+|\w+'\w+|#?\w+-?\w+|\w+[\*\$]+\w+?''', text)
            if tokens == []:
                print('\nNo tokens for text:\n', t)
                result.append(' ')
                continue
            for token in tokens:
                token = re.sub('\s', '', token)
                token = re.sub('\.', '', token)
                token = re.sub('\#', '', token)
                if token.lower() not in stop_words:
                    lemmatized_words.append(lemmatizer.lemmatize(token).lower())
            result.append(' '.join(lemmatized_words))
            progress_bar.update()
    return result

def spacy_lemmatizer(texts, nlp):
  result = []
  with tqdm(total=len(texts)) as progress_bar:
    for t in texts:
      lemmatized_texts = []
      doc = nlp(t)
      for token in doc:
        #if not sum([token.is_punct, token.is_space, token.is_stop, token.is_digit, token.like_num]):
        #  if token.lang_ == 'en':
        #    if token.like_url:
        #      lemmatized_texts.append('url')
        #    else:
        #      lemmatized_texts.append(token.lemma_.lower())
        lemmatized_texts.append(token.lemma_.lower())
      result.append(' '.join(lemmatized_texts))
      progress_bar.update()
  return result

def get_tags(texts, nlp):
  result = []
  with tqdm(total=len(texts)) as progress_bar:
    for t in texts:
      tags = []
      doc = nlp(t)
      for token in doc:
        tags.append(token.tag_)
      result.append(' '.join(tags))
      progress_bar.update()
  return result


def split_tokenizer(text):
    return text.split(' ')

def fit_catboost(X_train, X_test, y_train, y_test, catboost_params={}, verbose=100):
    learn_pool = Pool(
        X_train,
        y_train,
        text_features=['comment'],
        feature_names=list(X_train)
    )
    test_pool = Pool(
        X_test,
        y_test,
        text_features=['comment'],
        feature_names=list(X_train)
    )

    catboost_default_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'eval_metric': 'F1',
        'task_type': 'GPU'
    }

    catboost_default_params.update(catboost_params)

    model = CatBoostClassifier(**catboost_default_params)
    model.fit(learn_pool, eval_set=test_pool, verbose=verbose)

    return model

def spacy_lemmatizer(texts, nlp):
  result = []
  with tqdm(total=len(texts)) as progress_bar:
    for t in texts:
      lemmatized_texts = []
      doc = nlp(t)
      for token in doc:
        lemmatized_texts.append(token.lemma_.lower())
      result.append(' '.join(lemmatized_texts))
      progress_bar.update()
  return result

def get_ncapitals(not_lemmatized_texts):
  result = []
  for text in not_lemmatized_texts:
    capitals = 0
    for character in text:
      if character.isupper():
        capitals += 1
    result.append(capitals/len(text))
  return result

def get_nbadwords(lemmatized_texts, badwords):
    result = []
    for text in lemmatized_texts:
      n_badwords = 0
      tokens = text.split(' ')
      for w in badwords:
        if w in tokens:
          n_badwords += 1
      result.append(n_badwords/len(tokens))
    return result

def get_nuniquewords(lemmatized_texts):
  return [len(set(text.split(' ')))/len(text.split(' ')) for text in lemmatized_texts]

def get_list_of_special_characters(not_lemmatized_texts):
  result = set()
  punct = {"!", '"', "'", ')', ')', ',', '-', '.', ':', ';', '?', '`'}
  for text in not_lemmatized_texts:
    result.update(set(text))
  for character in result.copy():
    if sum([character.isalpha(), character.isspace(), character.isdigit(), (character in punct)]) == 1:
      result.discard(character)
  return result

def get_nspecial(not_lemmatized_texts, character_list):
  result = []
  for text in not_lemmatized_texts:
    nspecial = 0
    for character in text:
      if character in character_list:
        nspecial += 1
    result.append(nspecial/len(text))
  return result

def all_metrics(y_true, y_pred):
    print("ROC_AUC: {:.3f}\nPrecision: {:.3f}\nRecall: {:.3f}\nCohen-Kappa: {:.3f}\nF1: {:.3f}".format(roc_auc_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred),
                                                                                             cohen_kappa_score(y_true, y_pred), f1_score(y_true, y_pred)))
