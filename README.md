# MCPredictor
Experiment code for "Integrating Deep Event-Level and Script-Level 
Information for Script Event Prediction", EMNLP 2021

## 1. Corpus
Corpus can be found in LDC: 
https://catalog.ldc.upenn.edu/LDC2005T12 ,
since this dataset use documents from year 1994 to 2004, please use at least the second edition.

## 2. MCNC dataset
MCNC dataset processing code can be found here: 
https://mark.granroth-wilding.co.uk/papers/what_happens_next/ .

Please use python2.7 environment to run this code.

Please follow bin/event_pipeline/rich_docs/gigaword.txt to construct the dataset.

### 2.1 Modification of Granroth-Wilding's code

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
It is recommended to use absolute directory  ```#!<data-dir>/bin/run_py``` instead of ```#!../run_py```

It is recommended to use lxml engine in BeautifulSoup:
```soup = BeautifulSoup(xml_data, "lxml")```

#### 2.1.3 modify directories
Data directories in following files should be changed to user directory:
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
"tokenized.tar.gz" should be decompressed 
into the same directory.

## 2. Installation
Use command ```pip install -e .``` in 
project root directory.

Using ```pip install -r requirements.txt``` to
install dependencies.

Be sure to use python>=3.6 environment.

## 3. Preprocess
Use command ```experiments/preprocess.py``` to preprocess data.
Following arguments should be specified:
- ```--data_dir```: the directory of MCNC dataset
- ```--work_dir```: the directory of temp data and results

## 4. Training
### train mcpredictor:
```python experiments/train.py --work_dir <work_dir> --model_config config/mcpredictor.json --multi```

### train scpredictor:
```python experiments/train.py --work_dir <work_dir> --model_config config/scpredictor.json```

## 5. Testing
### test mcpredictor:
```python experiments/test.py --work_dir <work_dir> --model_config config/mcpredictor.json --multi```

### test scpredictor:
```python experiments/test.py --work_dir <work_dir> --model_config config/scpredictor.json```
