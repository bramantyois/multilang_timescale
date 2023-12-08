import os

import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors

import fasttext
from fasttext.util import download_model


def get_word2vec(target_directory):
    '''Stores vectors(around 3.7GB) at target_directory.
    Returns a Gensim Word2VecKeyedVectors object.'''

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print('Loading Word2Vec vectors...')
        w2v = api.load('word2vec-google-news-300')
        w2v.save(os.path.join(target_directory, 'kv'))
        print('============ Word2Vec: SAVED TO DISK ============')
    else:
    	print('Loading Word2Vec vectors...')
    	w2v = KeyedVectors.load(os.path.join(target_directory, 'kv'))
    	print('Done')
    return w2v


def get_fasttext(target_directory, model, language_id='en'):
    '''Stores vectors at target_directory (model 1/ 2).
    Returns a Gensim (Word2VecKeyedVectors/ FastTextKeyedVectors) object.
    
    Model 1 is only available in English, based on-
    https://arxiv.org/abs/1712.09405
    
    Model 2 (CBOW) is available in 157 languages(give language_id if using Model 2), based on-
    https://arxiv.org/abs/1802.06893'''

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    if model == 1:
        if not os.path.exists(os.path.join(target_directory, 'kv1')):
            print('Loading FastText vectors...')
            ft = api.load('fasttext-wiki-news-subwords-300')
            ft.save(os.path.join(target_directory, 'kv1'))
            print('Done')
            print('============ FastText Wiki News: SAVED TO DISK ============')
        else:
            print('Loading FastText vectors...')
            ft = KeyedVectors.load(os.path.join(target_directory, 'kv1'))
            print('Done')
            
    elif model == 2:
        if not os.path.exists(os.path.join(target_directory, language_id+'_kv2')):
            current_directory = os.getcwd()
            os.chdir(target_directory)
            
            print('Downloading vectors...')
            download_model(language_id, if_exists='ignore')
            print('Done')
            print('Loading FastText vectors...')
            ft = load_facebook_vectors('./cc.{}.300.bin'.format(language_id))
#             ft = fasttext.load_model('./cc.en.300.bin')  # fasttext Model
            print('Done')
    
            os.system('rm ./cc.{}.300.bin*'.format(language_id))        
            ft.save(language_id+'_kv2')
            print('============ FastText CBOW: SAVED TO DISK ============')
            os.chdir(current_directory)
        else:
            print('Loading FastText vectors...')
            ft = KeyedVectors.load(os.path.join(target_directory, language_id+'_kv2'))
            print('Done')

    return ft


if __name__ == '__main__':
    word2vec_directory = '../../Word2Vec_Pretrained/'
    fasttext_directory = '../../FastText_Pretrained/'

    w2v = get_word2vec(word2vec_directory)  # Word2VecKeyedVectors
    ft_english = get_fasttext(fasttext_directory, model=2, language_id='en')  # FastTextKeyedVectors
    ft_tamil = get_fasttext(fasttext_directory, model=2, language_id='ta')  # FastTextKeyedVectors
    
    print(w2v)
    print(ft_english)
    print(ft_tamil)
