CSCI57300	Data Mining
Homework 2  PCA Algorithm
Author: Jingyi Hui
Data:	09/30/2018

-----------------------------------------
List of Documents:
1. README.txt
2. JH_myKeans.py
3. iris.txt
4. Evaluation.pdf (Purity-based evaluation report)

-----------------------------------------
To run the program with iris data:
1. Login to Tesla and copy all the files under a directory;
2. Make the .py file executable, type:
	chmod +x JH_myKmeans.py
3. To run the program without centroid list file, type:
	./JH_myKmeans.py iris.txt 3	
4. To run the program without centroid list file, type:
	./JH_myKmeans.py iris.txt 3 centroidfile.txt
5. User can choose other data files and numbers as input. If the "centroidfile.txt" is part of the input, the program will check both the number of centroids in the file and the number of centroids user defined. If the numbers don't match, the program will show the error message and exit.
6. The console will display all the dataset and result information.
