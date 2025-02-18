### Datasets:

* **METABRIC** [1]: Consists of 1, 440 samples from breast cancer patients which are annotated by 8 subtypes based on [2]. We observe two modalities, namely the RNA gene expression data, and Copy Number Alteration (CNA) data. The dimensions of these modalities are 15, 709 and 47, 127 respectively. 

* **Reuters** [3]: Consists of 18, 758 documents from 6 different classes. Documents are represented as a bag of words, using a TFIDF-based weighting scheme. This dataset is a subset of the Reuters database, comprising the English version as well as translations in four distinct languages: French, German, Spanish, and Italian. Each language is treated as a different view. To further reduce the input dimensions we preprocess the data  with a truncated version of SV D, turning all input dimensions to 3,000. 

* **Caltech101-20** [4]: Consists of 2, 386 images of 20 classes. This dataset is a subset of Caltech101. Each view is an extract handcrafted feature, including Gabor feature, Wavelet Moments, CENTRIST feature, HOG feature, GIST feature and LBP feature. The creation of both Caltech101-20 and Caltech-5V-7 is due to the unbalance classes in Caltech-101.

* **VOC** [5]: Consists of 9, 963 image and text pairs from 20 different classes. Following the conventions by [6,7], 5,649 instances are selected to construct a two-view dataset, where the first and the second view is 512 Gist features and 399 word frequency count of the instance respectively.

* **Caltech-5V-7** [8]: Consists of 1,400 images of 7 classes. Same as Caltech101-20, this dataset is also a subset of Caltech101 and is comprised from the same views apart from the Gabor feature.

* **RBGD** [9]: Consists of 1,449 samples of indoor scenes image-text of 13classes. We follow the version provided in [6, 10], where image features are extracted from a ResNet50 model pretrained on the ImageNet dataset and text features from a doc2vec model pretrained on the Wikipedia dataset.

* **MNIST-USPS** [11]: Consists of 5,000 digits from 10 different classes (digits). MNIST, and USPS are both handwritten digital datasets and are treated astwo different views.

* **CCV** [12]: Consists of 6,773 samples of indoor scenes image-text of 20 classes. Flowing the convention in [13] we use the subset of the original CCV data. The views comprise of three hand-crafted features: STIP features with 5, 000 dimensional Bag-of-Words (BoWs) representation, SIFT features extracted every two seconds with 5,000 dimensional BoWs representation, and MFCC features with 4, 000 dimensional BoWs.

* **MSRCv1** [14] Consists of 210 scene recognition images belonging to 7 categories. Each image is described by 5 different types of features. 

* **Scene15** [15] Consists of 4,485 scene images belonging to 15 classes.




### References:


[1] Christina Curtis, Sohrab P Shah, Suet-Feung Chin, Gulisa Turashvili, Oscar M Rueda, Mark J Dunning, Doug Speed, Andy G Lynch, Shamith Samarajiwa, Yinyin Yuan, et al. The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups. Nature, 486(7403): 346–352, 2012.

[2] Sarah-Jane Dawson, Oscar M Rueda, Samuel Aparicio, and Carlos Caldas. A new genome-driven integrated classification of breast cancer and its implications. The EMBO journal, 32(5):617–628, 2013.

[3] Massih R Amini, Nicolas Usunier, and Cyril Goutte. Learning from multiple partially observed views-an application to multilingual text categorization. Advances in neural information processing systems, 22, 2009.

[4] Handong Zhao, Zhengming Ding, and Yun Fu. Multi-view clustering via deep matrix factorization. In AAAI, pp. 2921–2927, 2017.

[5] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision, 88:303–338, 2010.

[6] Daniel J Trosten, Sigurd Lokse, Robert Jenssen, and Michael Kampffmeyer. Reconsidering representation alignment for multi-view clustering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1255–1265, 2021a

[7] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.

[8] Delbert Dueck and Brendan J Frey. Non-metric affinity propagation for unsupervised image categorization. In 2007 IEEE 11th international conference on computer vision, pp. 1–8. IEEE, 2007.

[9] Chen Kong, Dahua Lin, Mohit Bansal, Raquel Urtasun, and Sanja Fidler. What are you talking about? text-to-image coreference. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3558–3565, 2014.

[10] Runwu Zhou and Yi-Dong Shen. End-to-end adversarial-attention network for multi-modal clustering. In CVPR, pp. 14619–14628, 2020.

[11] Arthur Asuncion and David Newman. Uci machine learning repository, 2007.

[12] Yu-Gang Jiang, Guangnan Ye, Shih-Fu Chang, Daniel Ellis, and Alexander C Loui. Consumer video understanding: A benchmark database and an evaluation of human and machine performance. In ICMR, pp. 1–8, 2011b.

[13] Zhaoyang Li, Qianqian Wang, Zhiqiang Tao, Quanxue Gao, and Zhaohua Yang. Deep adversarial multi-view clustering network. In IJCAI, pp. 2952–2958, 2019b.

[14] Wei Zhao, Cai Xu, Ziyu Guan, and Ying Liu. Multiview concept learning via deep matrix factorization. IEEE transactions on neural networks and learning systems, 32(2):814–825, 2020.

[15] Li Fei-Fei and Pietro Perona. A bayesian hierarchical model for learning natural scene categories. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), volume 2, pp. 524–531. IEEE, 2005