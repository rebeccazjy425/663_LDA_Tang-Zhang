The following is the intruction for using all files in this repository:

The repository contains the following items:
- "Report.ipynb": write ups for introduction and background of the project;
- "LDApackage": Package/Source code for both the Gibbs Sampling and EM algorithms;
- Data files:
    - "simulated.txt": simulated data;
    - "realdata.txt": real life data;
    - "stopword.txt": a list of stopwords to be removed from text while preprocessing;
- "setup.py": to initiate the "LDApackage" for use;
- "Simulated+Data.ipynb": test codes for simulated data and write ups for this example;
    - "VIEM_Sim_topwords.txt": results from VIEM on the simulated data;
    - "Gibbs_Sim_topwords.dat": results from Gibbs Sampling on the simulated data;
- "Real+Data.ipynb": test codes for real data and write ups for this example;
    - "VIEM_RD_topwords.txt": results from VIEM on the simulated data;
    - "Gibbs_RD_topwords.dat": results from Gibbs Sampling on the simulated data;
- "Comparison.ipynb": comparative studies with other algorithms and write ups for this part;
    - "simu": Unix Executable file for simulated data to be used in Mixture of Unigrams model;
    - "GibbsSamplingDMM.py", "pDMM.py": source code for Mixture of Unigrams model;
    - "output": folder for Mixture of Unigrams model output;
- "README.md": this file, instructions on the repository;
- "LICENSE": open source license.

***Write ups for each example and comparative studies, see the ipynb files of the specific examples.

1. To access the source code:
In the "LDApackage" folder, you will have access to the package with Gibbs Sampling method. You can also find the implementation with variational inference in "LDA_VIEM" folder.

2. To access test code and example:
All text files for examples are in the same directory as codes and write ups. Please see section above for the specific file names and contents.

3. To reproduce the results:

***Gibbs Sampling: all result files, after running the code, would be saved as files named "topwords.dat". Each time of re-running or switching data file, the previous results would be overwritten. To keep the former results, please make sure you rename or save the previous .dat file before re-running the code. If parameter values changed, please restart kernel and re-run all codes.

***Variational Inference + EM algorithm: all result files, after running the code, would be saved as files named "VIEM_RD_topwords.txt". Each time of re-running or switching data file, the previous results would be overwritten. To keep the results, please make sure you rename or save the previous .txt file before re-running the code.

4. To change parameters:

***Gibbs Sampling: to alter the number of topics/top words generated, go to "__init__.py" under "LDApackages" folder and change the parameter under the initialization function in the LDAModel object. self.K is the number of topics and self.twords is the number of top words under each topic.

***Variational Inference + EM algorithm: to alter the number of topics/top words generated, directly call the function LDA_VIEM(documents,num_topic,maxTopicWordsNum_show) from LDA_VIEM and change the parameters of 'num_topic' and 'maxTopicWordsNum_show' respectively.
