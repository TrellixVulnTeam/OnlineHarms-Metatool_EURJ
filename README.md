# OnlineHarms-Metatool

The goal of this project is to provide an out-of-the-box way to test whether a text is likely to be abusive or not using previously published classifiers trained on published annotated abusive speech datasets. 

Simply put, given a file of texts, the program generates several reports on whether each text is abusive or not. 

## Installation 

Simply use the venv included in the project named `MetatoolVenv`. 

## How to Use

1. Place one or more .csv files in folder `Dataset`. Each .csv file must have **_only_** one column named `text`. Each line should be one comment/text. 
2. Run `main.py`. This will use all available trained models to generate predictions (i.e., whether each text is likely to be abusive or not). 
3. Get the results from folder `Results`. `0` means `Not Abusive`, `1` means `Abusive`.  

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
- Support more classifiers
- Support more abusive speech datasets 
- Bug fixes

## Acknowledgements
