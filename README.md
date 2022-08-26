xStream Explanation
===============

Extension of [xStream](https://dl.acm.org/doi/pdf/10.1145/3219819.3220107) to extract explanations from the detected anomalies.

The *src* directory contains an object oriented adaptaiton of the original [implementation](https://github.com/cmuxstream/cmuxstream-core).

The C++ code is compiled into a shared object (.so) using [PyBind11](https://github.com/pybind/pybind11) and [C++Import](https://pypi.org/project/cppimport/).


Installation
------------

* **1** Install all the dependencies:
    > python3 -m pip install -r requirements.txt

* **2** Run *main.py* which will compile the C++ code into a shared object binary (.so):
    > python3 main.py

* **3** Copy the shared object binary (.so) to the same folder of the python script:
    > cp src/xstream.*.so .

* **4** Rerun *main.py*:
    > python3 main.py


Usage
------------
```
main.py [--K projection_size] 
        [--C number_of_chains] 
        [--D depth] 
        [--W window_size] 
        [-d dataset_path] 
        [-m {feature_count, average_score, statistical_test}] 
        [-n number_of_additional_features] 
        [-tp] 
        [-t {t, ks}] 
        [-p pvalue threshold]
```

* **Optional Arguments:**
```
--K                   Number of projection subspaces to be used by xStream
--C                   Number of half-space chains to be used by xStream
--D                   Depth of each half-space chains to be used by xStream
--W                   Window size to be used by xStream
-d, --data            Path for the dataset on which xStream will be used
-m, --mode            Anomaly explanation technique to be applied
-n, --noise           Number of random noise features to be added to the original data
-tp, --true_positive  Will use only true positives to evaluate the explanations performance, instead of all detected anomalies
-t, --test            Choose the statistical test to be performed
-p, --pValue          Define the limit pValue on the statistical test
```