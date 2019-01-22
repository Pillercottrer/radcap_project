# radcap_project
This is code for a project in medical school trying to automate descriptions of medical images. 
It is based on different implementations of the CNN-RNN design. Each implementation has one python file for the model, one file to prapare
to prepare the data in batches one file to train the network and one file to generate captions with the trained network. 
There is also code in the radcap_project/data_preprocessing/ folder for creating the vocabulary.

To train the network run the train file and to generate captions run the caption generation file. 

Implementations:
1. The CNN-RNN from the paper "Show and tell" https://arxiv.org/abs/1411.4555
  File 1(the model): radcap_project/cnn_rnn_models/cnn_rnn_vinalys/
  File 2(dataprep): radcap_project/cnn_rnn_models/cnn_rnn_vinalys/dataHandler.py
  File 3(train): radcap_project/train.py
  File 4(generate captions): radcap_project/sample.py
2. An attempt to improve this with visual attention https://arxiv.org/abs/1502.03044
  File 1(the model): radcap_project/cnn_rnn_models/cnn_rnn_xu/model_cnnrnn_attention_sgrvinod.py
  File 2(dataprep): radcap_project/cnn_rnn_models/cnn_rnn_xu/dataHandlerAttention.py
  File 3(train): radcap_project/train_attention.py
  File 4(generate captions): radcap_project/caption_beam.py
3. The design proposed in the paper "On the Automatic Generation of Medical Imaging Reports"https://arxiv.org/abs/1711.08195
  File 1(the model): radcap_project/cnn_rnn_models/cnn_rnn_hierarchical/model_hierarchical.py
  File 2(dataprep): radcap_project/cnn_rnn_models/cnn_rnn_hierarchical/dataHandlerHierarchical.py
  File 3(train): radcap_project/train_hierarchical.py
  File 4(generate captions): radcap_project/caption_beam_hierarchical.py
  
  
For questions you can contact me at:
Emil.henning@gmail.com
