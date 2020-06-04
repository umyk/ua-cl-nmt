# Uncertainty-Aware Curriculum Learning for Neural Machine Translation

## Requirements
- Python version >= 3.7.4
- Pytorch version >= 1.2.0

# Getting Started
**Data Preprocessing**

We use the standard validation and test sets provided in each translation task.

**Language Models**

The main language models used in this paper can be obtained from the following links.
KenLM https://github.com/kpu/kenlm 
Bert as LM https://github.com/xu-song/bert-as-language-model

**Data Uncertainty**

Follow these steps to calculate the data uncertainty and generate the data difficulty JSON file:

1.Calculate the perplexity of each sentence.

2.Sort from low to high.

3.According to your experimental requirements, divide the sorted data set into several bins.

4.Extract 1000 pairs of sentences from each bin to verify the model uncertainty for a certain stage.

5.Build data difficulty JSON file. This is an example of data difficulty JSON file for 4 bins. These numbers are the indices of the sentences in a data file, the train_set represents all the data of the training set, the esti_set represents the estimation set.

{

	"train_set": [
  
		[116518, 41568, 13049, ..., 39342, 23659, 76413], 
    
		[12051, 113004, 57498, ..., 51064, 47300, 47552], 
    
		[73186, 50806, 17741, ..., 94891, 55986, 44589],
    
		[69885, 114662, 32893, ..., 103985, 85597, 84899]
    
	],	
  
	"esti_set": [
  
		[28948, 87465, 7934, ..., 7839, 89179, 55998], 
    
		[297, 84844, 4712, ..., 112400, 105640, 47525], 
    
		[115014, 71806, 46151, ..., 41996, 43563, 95774], 
    
		[22106, 66255, 72142, ..., 16703, 45681,  5157]
    
	]
  
}

**Train**

An example of a training script could be found in the script folder. Most parameters are quite obvious. Some parameters need to be specially set are explained as follows：


--file_prefix	Specify the directory of dataset.

--difficulty_json Specify the path of data difficulty JSON file.

--fold_name	Specify the directory to store models.



