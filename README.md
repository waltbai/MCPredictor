# MCPredictor
Experiment code for:

Long Bai, Saiping Guan, Jiafeng Guo, Zixuan Li, Xiaolong Jin, and Xueqi Cheng.
"*Integrating Deep Event-Level and Script-Level Information for Script Event Prediction*", EMNLP 2021


## 1. Corpus
Corpus can be found in LDC: 
https://catalog.ldc.upenn.edu/LDC2011T07 ,
since this dataset use documents from year 1994 to 2004, please use at least the second edition.

## 2. MCNC dataset
MCNC dataset processing code can be found here: 
https://mark.granroth-wilding.co.uk/papers/what_happens_next/ .

Please use python2.7 environment to run this code.

Please follow  ```README.md``` and ```bin/event_pipeline/rich_docs/gigaword.txt``` to construct the dataset.

### 2.1 Modification of Granroth-Wilding's code
Please let me know if I forget any changes.

#### 2.1.1 modify ```bin/run```
Since some computer run in other languages, which may raise error
when using JMNL, please set system language to english:
```
java -classpath $BUILD_DIR:$DIR/../src/main/java:$DIR/../lib/* \
    -DWNSEARCHDIR=$DIR/../models/wordnet-dict \
    -Duser.language=en \
    $*
```

#### 2.1.2 modify ```bin/event_pipeline/1-parse/preprocess/gigaword/gigaword_split.py```
It is recommended to use absolute directory  ```#!<code-dir>/bin/run_py``` instead of ```#!../run_py```

It is recommended to use lxml engine in BeautifulSoup:
```soup = BeautifulSoup(xml_data, "lxml")```

#### 2.1.3 modify directories
Data directories in following files should be changed to user's data directory:
- ```bin/event_pipeline/config/gigaword-nyt```
- ```bin/event_pipeline/rich_docs/gigaword.txt```
- ```bin/entity_narrative/eval/experiments/generate_sample.sh```

#### 2.1.4 modify ```bin/event_pipeline/1-parse/candc/parse_dir.sh```
Change to :
```../../../run_py ../../../../lib/python/whim_common/candc/parsedir.py $*```

#### 2.1.5 modify unavailable URLs in ```lib/```
C&C tool: https://github.com/chbrown/candc

OpenNLP: http://archive.apache.org/dist/opennlp/opennlp-1.5.3/apache-opennlp-1.5.3-bin.tar.gz

Stanford-postagger: https://nlp.stanford.edu/software/stanford-postagger-full-2014-01-04.zip

#### 2.1.6 extract tokenized documents
Since original texts are needed,
```<data-dir>/gigaword-nyt/tokenized.tar.gz``` should be decompressed 
into the same directory.

#### 2.1.7 Build java files
Create a build directory:
```bash
mkdir build
cd build
vi build.sh
```

Content in ```build.sh```:
```bash
cd <code_root>/src/main/java/cam/whim/opennlp
javac -Djava.ext.dirs=<code_root>/lib -d <code_root>/build *.java
cd <code_root>/src/main/java/cam/whim/narrative/chambersJurafsky
javac -Djava.ext.dirs=<code_root>/lib -d <code_root>/build *.java
cd <code_root>/src/main/java/cam/whim/coreference/simple
javac -Djava.ext.dirs=<code_root>/lib -d <code_root>/build *.java
cd <code_root>/src/main/java/cam/whim/coreference
javac -Djava.ext.dirs=<code_root>/lib -classpath <code_root>/build -d <code_root>/build *.java
```


## 3. Installation
Use command ```pip install -e .``` in 
project root directory.

Use command ```pip install -r requirements.txt``` to
install dependencies.

Environment: python>=3.6.

## 4. Preprocess
Use command ```python experiments/preprocess.py --data_dir <data_dir> --work_dir <work_dir>``` to preprocess data.
Following arguments should be specified:
- ```--data_dir```: the directory of MCNC dataset
- ```--work_dir```: the directory of temp data and results

On my working platform, It takes about 7 hours to generate the single chain train set, 
and takes about 10 hours to generate the multi chain train set.
Please make sure that the process will not be interrupted. 

## 5. Training
### train mcpredictor:
```python experiments/train.py --work_dir <work_dir> --model_config config/mcpredictor-sent.json --device cuda:0 --multi```

### train scpredictor:
```python experiments/train.py --work_dir <work_dir> --model_config config/scpredictor-sent.json --device cuda:0```

## 6. Testing
### test mcpredictor:
```python experiments/test.py --work_dir <work_dir> --model_config config/mcpredictor-sent.json --device cuda:0 --multi```

### test scpredictor:
```python experiments/test.py --work_dir <work_dir> --model_config config/scpredictor-sent.json --device cuda:0```


## 7. Citation

If you find the resource in this repository helpful, please cite

```
@inproceedings{bai-etal-2021-integrating,
    title = "Integrating Deep Event-Level and Script-Level Information for Script Event Prediction",
    author = "Bai, Long  and Guan, Saiping  and Guo, Jiafeng  and Li, Zixuan  and Jin, Xiaolong  and Cheng, Xueqi",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.777",
    pages = "9869--9878",
}
```
