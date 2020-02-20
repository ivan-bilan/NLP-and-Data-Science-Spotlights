# NLP and Data Science GitHub Repository Spotlight
Daily spotlights of some underrated NLP and Data Science GitHub repositories.

#### Spotlight №1 (January 18, 2020)
Topic Modelling approach called CorexTopic: https://github.com/gregversteeg/corex_topic

CorexTopic, or to be more verbose, Anchored Correlation Explanation Topic Modeling is a form of topic modeling that you should use when you already more or less know the topics you are expecting to extract from the dataset. Using CorexTopic you can help the model make those topics more precise and elaborate by providing so-called anchor words. With this approach, your topic clusters will be more precise. Try it out, it's a huge time saver when it comes to Topic Modeling.

And since we are on the topic of Topic Modelling, also take a look at a similar approach called GuidedLDA: https://github.com/vi3k6i5/GuidedLDA

#### Spotlight №2 (January 19, 2020)
Keywords extraction library called YAKE: https://github.com/LIAAD/yake

When it comes to keyword extraction the first instinct of many NLP experts is to try TextRank or RAKE. If you want to expand your tool-set when doing keyword extraction, YAKE is a great way to get keywords quickly and without any training of ML models from any document. The repository also gives a really great overview of similar keyword extraction algorithms. Give it a try.

#### Spotlight №3 (January 20, 2020)
Transfer Learning library for NLP called FARM: https://github.com/deepset-ai/FARM

FARM builds upon the very successful Transformers package (https://github.com/huggingface/transformers) from Hugging Face and incorporates many existing models like BERT or XLNet. With FARM you can pre-train them easily for any downstream NLP tasks. FARM is great for some fast prototyping and proof-of-concept to show your PM that transfer learning is the way to go.

#### Spotlight №4 (January 21, 2020)
Today's pick is an Entity Matching approach that allows you to pre-train a Deep Learning model on any labeled data you might have: https://github.com/anhaidgroup/deepmatcher

Entity Matching has usually been done with a lot of hand-crafted features, Deep Matcher is one of the few DL based approaches to Entity Matching that actually work out of the box. If you have to match two databases and eliminate the duplicates, DeepMatcher is a great starting point.

#### Spotlight №5 (January 22, 2020)
Seq2seq library Headliner: https://github.com/as-ideas/headliner

Originally developed at Axel Springer by Christian Schäfer and Dat Tran, this library is a great way to train and deploy your seq2seq models. It includes Transformer based models that can be used for summarization. Originally created to generate a headline for a piece of news, it can be used for many other tasks as well. Headliner is a great tool to try out for anyone working on summarization or wants to expand their understanding of what the Transformer architecture is capable of. 

#### Spotlight №6 (January 23, 2020)
NLP library that incorporates many Deep Learning-based models into one easy to use package called gobbli: https://github.com/RTIInternational/gobbli

Its motto is 'Deep learning with text doesn't have to be scary.', and it fulfills the promise by delivering an easy to use interface that covers many NLP approaches. You can use this one for quick prototyping with ease, try it out!

#### Spotlight №7 (January 24, 2020)
Compendium of all latest impactful NLP papers from the top NLP conferences: https://github.com/soulbliss/NLP-conference-compendium

It comprises links to papers that have won the best paper and best demo awards at ACL and EMNLP conferences for the past few years and also links to various tutorials and eventually all other accepted papers. Great way to track the progress in NLP.

#### Spotlight №8 (January 25, 2020)
A library that enables data scientists and data engineers to write data related tests faster. It's called "great expectations": https://github.com/great-expectations/great_expectations

It has a collection of ready to use testing functions that will test you tabular data for various potential pitfalls that you as a developer might not have accounted for right away. You should try them out and add them to your integration tests to avoid any unpleasant surprises.

To go with that, also try out snorkel (https://github.com/snorkel-team/snorkel) to quickly generate some test data.

#### Spotlight №9 (January 26, 2020)
A package that let's you automatically extend your textual training data: https://github.com/makcedward/nlpaug

With "nlpaug" you can automatically regenerate a sentence and replace various words in it with synonyms, antonyms, misspelled varients and more. You can also generate similar sentences and change the context within them using Transformer based generator models. Great way to artificially augment your NLP data with meaningful examples and minimal effort.

#### Spotlight №10 (January 27, 2020)
A package for cleaning tabular data called PyJanitor: https://github.com/ericmjl/pyjanitor

PyJanitor gives you access to a lot of cleaning functions that can make your DataFrames more consistent. You automatically remove empty rows and columns, identify duplicate entries, encode categories as categorical data for faster processing and do various forms of data conversions, all within one package.

#### Spotlight №11 (January 28, 2020)
An implementation of a Plug and Play Language Models (PPLM) from Uber: https://github.com/uber-research/PPLM

If you work a lot with text generation and you are having problems running GPT-2 in production, your best bet is to try PPLM, which is significantly smaller but still provides great performance. PPLM's other strength is its customizability. Definitely try it out.

#### Spotlight №12 (January 29, 2020)
A package that incorporates almost all imaginable readability index functions, it's called "textstat": https://lnkd.in/dX6WkEZ 

With "textstat" you can check how complex any given text is. The package includes Flesch Kinkaid, SMOG, Gunning Fog and many other readability index functions. Extremely useful if you work on Authorship Attribution or Profiling, or also Author Obfuscation.

#### Spotlight №13 (January 30, 2020)
A closed domain question answering system called cdQA: https://github.com/cdqa-suite/cdQA

cdQA uses BERT to create a question answering system for various specific domains (i.e. Finance, Medicare etc.) It would be a great addition to your Intranet search queries to find the information you are looking for from within the company documents.

#### Spotlight №14 (January 31, 2020)
An NLP toolkit that is built around sentence understanding tasks, its called jiant: https://github.com/nyu-mll/jiant

Jiant will help you fast and effectively pre-train transfer learning models for various multitask learning problems. It has some great built-in benchmarks and baselines for tasks like GLUE. And, in my view, it's strongest suit is the focus of the library on building easy to use sentence-level models. Its a must-try for any NLP practitioner.

#### Spotlight №15 (February 1, 2020)
A new approach from the NLP community called FitBERT: https://github.com/Qordobacode/fitbert

With FitBERT you can fill in the blanks within a sentence, namely you can mask out any word in the sentence and if you provide a list of replacement options for that word, FitBERT will select the best one by taking the context of the whole sentence into account. Works really great when also selecting the correct grammatical form of a word, so it can be, for example, used as an addition to existing grammar correction tools. How would you use FitBERT? 

#### Spotlight №16 (February 2, 2020)
Apackage that merges the functionalities of one of the best machine learning libraries, scikit-learn, with one of the arguably best deep learning libraries, PyTorch. Its name is skorch: https://github.com/skorch-dev/skorch

"Skorch" provides the best of both worlds, you can use PyTorch neural networks while loading data, running grid search, setting up model checkpoints and more using the all-to-familiar sklearn interface. Amazing library for fast experimentation for both ML and DL!

#### Spotlight №17 (February 3, 2020)

A deep learning library that can be used to train Transformer models quickly and effectively, it's called fast-bert: https://github.com/kaushaltrivedi/fast-bert

The inspiration for fast-bert comes from fast.ai since the interface of this library is made easy and accessible to beginners. Besides BERT, the library also includes recent additions to the Transformer family of Neural Networks, like ALBERT, DistilROBERTA and a lot more. You can use it to create a text classification model, pre-train language models and even directly deploy the final models to AWS Sagemaker.

#### Spotlight №18 (February 4, 2020)

A library for scalable NLP, called Spark-NLP: https://lnkd.in/dv2zDEg

As you might have guessed, it is built to run on Spark in a distributed environment. It can run on a cluster of CPUs or GPUs and it also easily configurable for both in-house clusters and the cloud (i.e. S3). Recent release includes faster implementation of word embeddings, support for ELMO models and the Universal Sentence Encoder. Put that 1000 CPUs on your inhouse cluster to use with Spark-NLP.

#### Spotlight №19 (February 5, 2020)

A library that can extract structured information from any kind of sentence: https://github.com/snipsco/snips-nlu

Snips is a great library for Natural Language Understanding. It can transform any text into an automatically structured JSON format. It can also be used to detect the intent that can then be passed to your chatbot. It can be trained on your own data and the library also includes an official benchmark. It performs better than most of the NLU libraries out there. If you are working on chatbots or a product that needs to extract structured data from text, snips is a great library to try.

#### Spotlight №20 (February 17, 2020)

Today's pick is a library that makes usage of Transformer-based models easier, its called Happy Transformer: https://github.com/EricFillion/happy-transformer

With Happy Transformer, you can train XLNET, BERT and ROBERTA; predict the next word in a sentence; show the probability of the next sentence in the paragraph or train a question answering system. Amazing work by Eric Fillion and his team, contributions are always welcome.

#### Spotlight №21 (February 18, 2020)

A set of scripts that will allow you to pre-train a GPT2 text generation model on your own dataset easily, it's called TextAugmentation-GPT2: https://github.com/prakhar21/TextAugmentation-GPT2

You can use TextAugmentation-GPT2 to quickly generate any kind of sentences within your own domain. As the name suggests, it's a great way to do some text augmentation to extend your textual dataset.

#### Spotlight №22 (February 19, 2020)

A package that helps researchers train production ready PyTorch models without writing too much boilerplate code, it's called PyTorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning

PyTorch Lightning is a framework that includes a lot of engineering best practices related to creating production-ready DL models. As a researcher, you can concentrate more on the research code while PyTorch Lightning will abstract away many of the engineering considerations.

#### Spotlight №23 (February 20, 2020)

A viewer of Neural Network models called Netron: https://github.com/lutzroeder/netron

With netron you can import your existing model for inspection and visualization. Netron supports a wide array of various model formats, starting from Keras and PyTorch models up to pickled sklearn models. Netron is a great way to see the structure of your model in more detail.

### Follow me for more content like this:
- LinkedIn: https://www.linkedin.com/in/ivan-bilan/
- Twitter: https://twitter.com/DemiourgosUA
- GitHub: https://github.com/ivan-bilan
- Medium (for non-tech content): https://medium.com/@demiourgosua/
