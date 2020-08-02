# Ensuring Fairness in Group Recommendations by Rank-Sensitive Balancing of Relevance

This repository contains the source code to reproduce the results in our paper "Ensuring Fairness in Group Recommendations by Rank-Sensitive Balancing of Relevance" in the Proceedings of RecSys 2020.
## Authors
Mesut Kaya

Derek Bridge

Nava Tintarev

## Dependencies
The main python module dependencies are:
- numpy
- pandas
- sklearn
- json

**Note**: We have used Python 3.7.3 :: Anaconda 4.1.1 (64-bit). Some of our python module codes are adapted from "[vae_cf](https://github.com/dawenl/vae_cf)", for preprocessing the datasets.

The main Java module dependencies can be found in the pom.xml file. Our implementation is based on "[RankSYS Framework](https://github.com/RankSys/RankSys)". The code in src/mf is also from RankSys, we only added a random seed for the initialization of the latent factors of Matrix Factorization algorithm for reproducability. 

**Note**: We have tested the Java code with both Java(TM) SE Runtime Environment (build 12.0.1+12) and (build 13+33) in our experiments. 


## Preprocessing the dataset and creating sythetic groups

The python codes are under GFAR_python folder. The ratings files are under the data folder. For MovieLens dataset, data/ml1m/ratings.dat file; for KGRec-music dataset data/kgrec/implicit_lf_dataset.csv file.
Run the following python scripts:
```
python movielens.py
python kgrec_dataset.py
```

under both data/ml1m/ and folders data/kgrec/ 5 folders are created (for 5-fold cross validation in the experiments). Note that, for both datasets, original user and item ids are mapped into integer ids starting from 0. We us the mapped ID in the remainder of the experiments. (Note that, you may have some warnings while running above scripts.) 

After running the above scripts, to create synthetic groups run following:

```
python create_group.py ML1M
python create_group.py KGREC
```



The Java codes for GFAR are under GFAR_java folder.


## Compiling by Maven and running the code!
Change your working directory to GFAR_java. Then the classes can be compiled and run as follows (by replacing your JAVA_HOME with /usr/java/default/ below): 

```
JAVA_HOME=/usr/java/default/ mvn clean compile
export MAVEN_OPTS="-Xmx16000M"
mvn exec:java -Dexec.mainClass="gfar.GenerateIndividualRecommendations"
mvn exec:java -Dexec.mainClass="gfar.AggregateRecommedations"
mvn exec:java -Dexec.mainClass="gfar.RunGFAR"
mvn exec:java -Dexec.mainClass="gfar.RunGreedyAlgorithms"
```

Running the above code will generate the group recommendations for both datasets for the group size m = 8 (the results for which are shown in Table 3 and Table 4). If you want to generate recommendations for the other group sizes as well, it is possible to change the line:

```
int[] groupSize = {8};
```

with the following:

```
int[] groupSize = {2,3,4,5,6,7,8};
```
in the AggregateRecommedations, RunGFAR and RunGreedyAlgorithms java files. 

To generate the results shown in Tables 3 and 4, finally, run the following:

```
mvn exec:java -Dexec.mainClass="gfar.FairnessCollectResults"
```
**NOTE** Do not forget to replace jdk path instead of /usr/java/default above. 
