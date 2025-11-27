# Twitter Bot Detection - Kaggle Competition

A machine learning project for binary classification of Twitter bot accounts using ensemble methods and transformer-based feature augmentation.

## Project Overview

This project implements a bot detection pipeline that combines:

- **Traditional ML models**: XGBoost, CatBoost, and LightGBM
- **Deep learning augmentation**: DistilBERT-based feature generation
- **Advanced feature engineering**: Log transformations, interaction features, and behavioral ratios

**Final Performance**: Cross-Validation AUC of **0.9480**

## Repository Structure

```
twitter_bot_kaggle_comp/
├── pipeline_final.ipynb    # Complete ML pipeline with DistilBERT training
├── Report.pdf             # Detailed project report and analysis
├── requirements.txt       # Python dependencies
└── data/                 # Data directory (not included in repo)
    ├── train.csv
    ├── test.csv
    ├── train_bot_prob_from_desc.csv
    ├── test_bot_prob_from_desc.csv
    ├── train_bot_prob_from_cat.csv
    ├── test_bot_prob_from_cat.csv
    └── submission-{auc}.csv
```

### Performance Metrics

| Model        | Validation AUC |
| ------------ | -------------- |
| XGBoost      | 0.9480         |
| CatBoost     | 0.9472         |
| LightGBM     | 0.9508         |
| **Ensemble** | **0.9528**     |

## Development Notes

### Iterative Testing

The notebook includes a validation split approach (commented out as "DO NOT RUN") for faster iteration during development:

- 80/20 train/validation split
- Allows rapid testing without full cross-validation
- Helps prevent overfitting during experimentation

### Memory Management

DistilBERT training includes explicit GPU memory clearing:

```python
del model, trainer
torch.cuda.empty_cache()
```

This prevents OOM errors during 5-fold cross-validation.

## Project Context

This was developed for a Kaggle competition focused on Twitter bot detection. The approach emphasizes:

1. **Feature quality over quantity** - Deep feature engineering from user metadata
2. **Ensemble diversity** - Combining tree-based models with different strengths
3. **Transfer learning** - Leveraging pre-trained language models for text features
4. **Robust validation** - 5-fold stratified cross-validation throughout
