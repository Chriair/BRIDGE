# BRIDGE: Biological Evidence Refinement and Heterogeneous Dynamic Gating for Gene Regulatory Networks

------

## 1. Environment Setup

```
conda env create -f environment.yml
conda activate bridge
```

------

## 2. Dataset Organization (IMPORTANT)

### 2.1 Benchmark Dataset Structure

Your benchmark datasets **must follow this exact structure**:

```
Benchmark Datasets/
├── Non-Specific Dataset/
│   └── hESC/
│       └── TFs+1000/
│           ├── BL--ExpressionData.csv
│           ├── BL--network.csv
│           ├── Label.csv
│           ├── TF.csv
│           └── Target.csv
├── Specific Dataset/
└── STRING Dataset/
```

- `BL--ExpressionData.csv` : gene × cell expression matrix
- `Label.csv` : ground-truth TF–target pairs
- `TF.csv` / `Target.csv` : TF and gene index lists

------

### 2.2 Train / Validation / Test Split Structure

After running the split script, the following directory will be created:

```
Data/
├── Non-Specific/
│   └── hESC 1000/
│       ├── sample1/
│       │   ├── Train_set.csv
│       │   ├── Validation_set.csv
│       │   └── Test_set.csv
│       ├── sample2/
│       ├── sample3/
│       ├── sample4/
│       └── sample5/
```

Each `sampleX` corresponds to one random split.

------

## 3. Generate Train / Validation / Test Sets

Run the following command **once per dataset**:

```
python Train_Test_Split.py \
  --data hESC \
  --net Non-Specific \
  --num 1000
```

This generates:

```
Data/Non-Specific/hESC 1000/sample1/
├── Train_set.csv
├── Validation_set.csv
└── Test_set.csv
```

------

## 4. Train and Evaluate BRIDGE

Use **one sample at a time** (e.g. `sample1`):

```
python BRIDGE.py \
  --net non_specific \
  --cell_type hESC \
  --tf_num 1000 \
  --sample sample1
```