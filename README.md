## Versioning

 - python 3.10.4
 - [pip requirements](requirements.txt)

## 1. Build the subcorpus desc. in section 3

    Note: amend the [config.json file to point to the compressed reddit dump files](config.json)

```
python3 get_seeded_mp.py
```

## 2. Topic analysis 

```
python3 topic_model.py
```

## 3. Examine topics wrt., most prominent ngrams

Refer to (jupyter notebook)[analysis_2015_topic.ipynb]

## 4. RQ1 analysis / section 4.1 / section 5.1

Refer to (jupyter notebook)[analysis_2015_topic_seeds.ipynb]

## Causal Models via PyMC

### wrt., people group == black seed ngrams

```
python3 SCM/black/S1.py
python3 SCM/black/S2.py
python3 SCM/black/S3.py
```

### wrt., people group == derogatory black seed ngrams

```
python3 SCM/derogatory_black/S1.py
python3 SCM/derogatory_black/S2.py
python3 SCM/derogatory_black/S3.py
```

## 5. RQ2 analysis: intervening on the causal models

refer to [jupyter notebook](causal_models.ipynb)

