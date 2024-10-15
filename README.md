## An adaptive anomaly detection model for time series decomposition of fusion spatiotemporal features of non-stationary data (FSTD-AD)

## Get Started

1. Install Python 3.11, PyTorch 2.2.1.

```
pip install -r requirements.txt
```

2. Download data.
3. Train and evaluate. You can reproduce the experiment results as follows:

```
bash ./script/run.sh
```



### Download data set

#### SWaT and WADI
SWaT and WADI datasets can be obtained by filling out the following form:

https://docs.google.com/forms/d/1GOLYXa7TX0KlayqugUOOPMvbcwSQiGNMOjHuNqKcieA/viewform?edit_requested=true



#### PSM

The dataset can be downloaded at:

https://github.com/eBay/RANSynCoders/tree/main/data



#### SMD

The dataset can be downloaded at:

https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset



#### MSL and SMAP

The dataset can be downloaded in the following ways

labeled_anomalies.csv: Data processing and data separation between the two spacecraft depend on this file


```
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip
wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

下载数据集放进对应文件夹，运行make_pk.py
