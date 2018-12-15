

1. Run genLabelTxt/genLabelTxt.ipynb to genrate two files needed for finetuning
    input : (1) a csv with photo_id and label (in the same folder)
            (2) path to test images (whose photo_id exists in the csv)
    output: (1) ../finetune_alexnet_with_tensorflow/train_yelp.txt
            (2) ../finetune_alexnet_with_tensorflow/val_yelp.txt

2. Use finetune_alexnet_with_tensorflow/finetune.py to train the model

3. Test using:
   (1) alexnet_test.ipynb
   This tests the accuracy of a labeled test set
   input: a csv with photo_id and labels
          a folder with labeled images specified by photo_is above
   *TO DO: add visulaization of conv layers
   
   (2) alexnet_business.ipynb
   This is used to visulalize real-time results for a particular business
   input: images from a business
   

