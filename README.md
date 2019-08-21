# EmotiKLUE #

## Introduction ##

EmotiKLUE is a submission to the [WASSA 2018 Implicit Emotion Shared
Task](http://implicitemotions.wassa2018.com/). The aim of the shared
task is to predict one of six emotions (anger, disgust, fear, joy,
sadness, surprise) from the context of masked emotion words as in the
following example:

> My dad was [#TRIGGERWORD#] that I washed his car so he gave me money
> to buy snacks 😢

EmotiKLUE tackles this task by learning independent representations of
the left and right contexts of the masked emotion word and by
combining those representations with an LDA topic model.

The system is described and evaluated in greater detail in Proisl et
al. (2018).


## Usage ##

For information on how to train, retrain or test a model or on how to
use it for prediction, see the help messages of the corresponding
subcommands:

    emotiklue.py {train,retrain,test,predict} -h


## References ##

  * Proisl, Thomas, Philipp Heinrich, Besim Kabashi, and Stefan
    Evert. 2018. “EmotiKLUE at IEST 2018: Topic-Informed
    Classification of Implicit Emotions.” In *Proceedings of the 9th
    Workshop on Computational Approaches to Subjectivity, Sentiment
    and Social Media Analysis (WASSA)*, 235–242. Brussels: Association
    for Computational Linguistics.
    [PDF](http://aclweb.org/anthology/W18-6234).
  
        @InProceedings{Proisl_et_al_IEST:2018,
          author    = {Proisl, Thomas and Heinrich, Philipp and Kabashi, Besim and Evert, Stefan},
          title     = {{EmotiKLUE} at {IEST} 2018: {T}opic-Informed Classification of Implicit Emotions},
          booktitle = {Proceedings of the 9th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis},
          year      = {2018},
          pages     = {235--242},
          address   = {Brussels},
          publisher = {Association for Computational Linguistics},
          url       = {http://aclweb.org/anthology/W18-6234},
        }
