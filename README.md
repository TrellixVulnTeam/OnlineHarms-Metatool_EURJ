# Online Harms: A Meta-Tool for Abusive Speech Detection

The goal of this project is to provide an out-of-the-box way to test whether a text is likely to be abusive or not using published classifiers trained on various annotated abusive speech datasets. 

Given a file of texts, the program generates reports on whether each text is abusive or not based on the available trained models. It also generates the results of an voting ensemble classifier. 

## Installation 

#### Method 1:

1. Clone the project by typing the following in your command line: `git clone https://github.com/amittos/OnlineHarms-Metatool.git`. 
2. Change directory: `cd OnlineHarms-Metatool/`
3. Use the virtual environment provided by running: `source MetatoolVenv/Scripts/activate`
4. Run the program by typing `python main.py`

#### Method 2:

Download the Docker version from [here](). 

## How to Use

1. Place one or more .csv files in folder `Dataset`. Each .csv file must have **_only_** one column named `text`. Each line should be one comment/text. 
2. Run `main.py`. This will use all available trained models to generate predictions (i.e., whether each text is likely to be abusive or not). 
3. Get the results from folder `Results`. `0` means `Not Abusive`, `1` means `Abusive`.  

## Pull Requests

Feel free to create pull requests for fixing bugs or adding new trained models to the project. 

#### How to add new models: 

If you are using one the already supported classifiers to train a model based on a new dataset, then simply use the `train()` function of the classifier to generate a model. Name the model as follows: `[Name of Classifier] + [Name of Dataset] + '.model'`, e.g., `DavidsonFounta.model`. Then place the model in the folder `Models` of the classifier you are using.

If you are using a new classifier, then make sure that the structure followed is the same as with the rest of the classifiers. Specifically:
- Inside folder `Classifiers` create a folder named after your classifier, e.g., `MyClassifier`.
- Inside folder `MyClassifier` create a script called `MyClassifierClassifier.py`. The script must have a function `train()` and a function `test()` similarly to the rest. 
- Name the model as follows: `[Name of Classifier] + [Name of Dataset] + '.model'`, e.g., `MyClassifierMyDataset.model`.
- Inside folder `MyClassifier` create a folder named `Models`. 
- Add your generated model into folder `Models`. 
- Add your classifier and dataset in the lists of `main.py`
- All done! 

## Classifiers Supported

- Davidson, Thomas, et al. "Automated hate speech detection and the problem of offensive language." Eleventh international aaai conference on web and social media. 2017.
- Wulczyn, Ellery, Nithum Thain, and Lucas Dixon. "Ex machina: Personal attacks seen at scale." Proceedings of the 26th International Conference on World Wide Web. 2017.
- Pelicon, Andraž, Matej Martinc, and Petra Kralj Novak. "Embeddia at SemEval-2019 Task 6: Detecting hate with neural network and transfer learning approaches." Proceedings of the 13th International Workshop on Semantic Evaluation. 2019.

## Datasets Supported

- Davidson, Thomas, et al. "Automated hate speech detection and the problem of offensive language." Eleventh international aaai conference on web and social media. 2017.
- Founta, Antigoni Maria, et al. "Large scale crowdsourcing and characterization of twitter abusive behavior." Twelfth International AAAI Conference on Web and Social Media. 2018.
- de Gibert, Ona, et al. "Hate speech dataset from a white supremacy forum." arXiv preprint arXiv:1809.04444 (2018).
- Qian, Jing, et al. "A Benchmark Dataset for Learning to Intervene in Online Hate Speech." Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2019.
- Toxic Comment Classification Challenge, https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/
- Waseem, Zeerak, and Dirk Hovy. "Hateful symbols or hateful people? predictive features for hate speech detection on twitter." Proceedings of the NAACL student research workshop. 2016.
- Wulczyn, Ellery, Nithum Thain, and Lucas Dixon. "Ex machina: Personal attacks seen at scale." Proceedings of the 26th International Conference on World Wide Web. 2017.
- Zampieri, Marcos, et al. "Predicting the Type and Target of Offensive Posts in Social Media." Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2019.

## Future Work
- Support more trained models
- Add a webpage interface 
- Bug fixes

## Acknowledgements

The development of this tool was supported by EPSRC Grant Ref: EP/T001569/1 for “Artificial Intelligence for Science, Engineering, Health and Government”, and particularly the “Tools, Practices and Systems” theme via the “Detecting and Understanding Harmful Content Online: A Metatool Approach” project.