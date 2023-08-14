## [Kaggle Competition] FeedBackPrize - English Language Learning
___
### Final Score
- Private Score: 0.436454
- Rank: 367/2654, 13.8%

___
### Competition Summary
-  To assess the language proficiency of 8th-12th grade English Language Learners (ELLs). Utilizing a dataset of essays written by ELLs will help to develop proficiency models that better supports all students.
___
### Strategy
#### **[Various Model Ensemble]**

![Modeling Overview](/assets/modeling_overview.png)

**Type 1. From Meta-Pseudo Label, Make Various Model**  
* **Make Best Single Fine-Tuned Model from Competition Data**
    - Backbone: DeBERTa-V3-Large
    - Pooling: MeanPooling
    - Loss: SmoothL1Loss
* **Make Student Model from Meta-Pseudo Labeling**
    - Un-Supervised Dataset: FBP1, FBP2 Competition Dataset
    - StudentModel Backbone: DeBERTa-V3-Large
    - Loss: WeightedMCRMSE  
        (For transfer FBP3 data's distribution to FBP1, FBP2)
    - Metrics: MCRMSE from FBP3 data
* **Fine-Tuned by FBP3 data with Student Model**
    - Backbone: student model with trained by Meta-Pseudo Label 
    - Various Pooling: MeanPool, GEMPool, WeightedLayerPool, AttentionPool
    - Various Sequence Length: 512, 756, 1024, 1536(max length of data)
    - Loss: SmoothL1Loss
    - Metrics: MCRMSE
    - Apply Freeze & Re-Init Encoders
    - Apply Layer Wise Learning Rate Decay

**Type 2. Only Labeled Data, Make Various Model**  
* **Make various Fine-Tuned Model from Competition Data**
    - Backbone: DeBERTa-V3-Large
    - Various Pooling: MeanPool, GEMPool, WeightedLayerPool, AttentionPool
    - Various Sequence Length: 512, 756, 1024, 1536(max length of data)
    - Loss: SmoothL1Loss
    - Metrics: MCRMSE
    - Apply Freeze & Re-Init Encoders
    - Apply Layer Wise Learning Rate Decay

**Type 3. Add RAPID's Support Vector Regressor Head**  
* **Model 1.Extract Embedding with Pretrained Model NOT Fine-Tuned with Competition Data**
    - Backbone: DeBERTa-V3-Large
    - Various Pooling: MeanPool, GEMPool, WeightedLayerPool(two different verison), AttentionPool
    - Various Sequence Length: 640
    - Concatenate Embedding from Five different Pooling
    - Regression with RAPID's SVR

* **Model 2.Extract Embedding with Pretrained Model Fine-Tuned with Competition Data**
    - Backbone: Fine-Tuned DeBERTa-V3-Large
    - Various Pooling: MeanPool, GEMPool, WeightedLayerPool(two different verison), AttentionPool
    - Various Sequence Length: 640
    - Concatenate Embedding from Five different Pooling
    - Regression with RAPID's SVR

___
# Reference  
##### https://www.kaggle.com/competitions/mayo-clinic-strip-ai/overview
##### https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/351577
##### https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609
##### https://arxiv.org/abs/2003.10580
##### https://arxiv.org/pdf/1904.12848v6.pdf
##### https://arxiv.org/abs/1711.02512
##### https://arxiv.org/abs/1810.04805
___