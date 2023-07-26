# Hot-dog-Not-hot-dog-Classifier

A binary classifier to test whether an image belongs to the "hot dog" class or the "not hot dog" class, as seen on HBO's [*Silicon Valley*](https://www.bing.com/videos/riverview/relatedvideo?&q=hot+dog+not+a+hot+god+silicon+valley&&mid=162A96163FFFB5F6FCB1162A96163FFFB5F6FCB1&&FORM=VRDGAR).

![Not hot dog](images/not_hot_dog.jpeg)

# Hot Dog Classification with Transfer Learning

This project uses transfer learning to build a binary classifier that can distinguish between images of hot dogs and images of "not hot dogs." Four different pre-trained models are used: InceptionV3, MobileNetV2, ResNet50, and VGG16. (See below for paper references.)

The [Hot dog - Not hot dog](https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog) Kaggle dataset is used for training and validation.

The pre-trained models are loaded with weights from the ImageNet dataset and fine-tuned on the training data. Initially, all layers of the base model are frozen, meaning their weights are not updated during training. After an initial training phase of 5 epochs, however, the base model is set to be trainable and fine-tuned for an additional 5 epochs (from layer 0 onward) with a lower learning rate. (For clarification, see the `notebook` directory to find my notebook for the MobileNetV2 model.)

After training, the models achieve the following validation accuracies:

| Model       | Validation accuracy (after 5 epochs with pre-trained ImageNet weights) | Validation accuracy (after 5 epochs with fine-tuning from layer 0 onward) |
|-------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------|
| InceptionV3 | 0.9397                                                                | 0.9397                                                                    |
| MobileNetV2 | 0.9095                                                                | 0.9146                                                                    |
| ResNet50    | 0.8894                                                                | 0.9045                                                                    |
| VGG16       | 0.8141                                                                | 0.8141                                                                    |


Overall, it seems that the InceptionV3 model has the best performance. Hot dog! 

Paper references:
- [InceptionV3](https://arxiv.org/pdf/1409.4842.pdf)
- [MobileNetV2](https://arxiv.org/pdf/1704.04861.pdf)
- [ResNet50](https://arxiv.org/pdf/1512.03385.pdf)
- [VGG16](https://arxiv.org/pdf/1409.1556.pdf)


