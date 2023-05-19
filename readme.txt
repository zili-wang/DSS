####ANY ADJUSTMENT OF THE SOFTWARE SHOLD BE NOTIFIED TO WANGZILIHIT@OUTLOOK.COM####

#STEP 1: Install python 
You can download python free from https://www.python.org/
The recommended version is python 3.6.1(64-bit)

#STEP 2: Install Acaconda
You can download Acaconda free from https://www.anaconda.com/
Download the version of Acaconda which matches your installed python version.

#STEP 3: prepare the running environment
(1)Open CMD
(2)Type code:
  py -m pip install pandas
  py -m pip install sklearn
  py -m pip install matplotlib
  py -m pip install seaborn
  py -m pip install pydotplus
  py -m pip install IPython
  py -m pip install graphviz
  py -m pip install xlrd
  py -m pip install openpyxl
  py -m pip install PyQt5
(3)Install graphviz in the following steps:
	1.Open Anaconda Prompt
	2.type code:
		conda install -c anaconda graphviz 
(4)Set up graphviz as the environment variable in the following steps:
	1.Open Control panel
	2.Click System and Security
	3.Click System
	4.Click Advanced system settings
	5.Click Environment Variables
	6.Choose Path, and click Edit
	7.Click New, add code "The location of Acaconda+"\Library\bin\graphviz 
	for example, if you install Anaconda in C:Users\
	the code for creating the new path of environment variable is C:Users\Anaconda3\Library\bin\graphviz
	8.After setting up graphviz as the environment variable, you may need to restart python to get it work
	
  
#STEP 4(Only for software developer): prepare the input data
(1) Ensure the input data is a .xlsx file 
(2) Ensure the name of the input data file is DTC.xlsx
(2) Ensure the input data is in the same document where Main.py is located

#STEP 5: run the software 
(1)Open IDLE (python 3.6 64-bit). If your intalled pyhton version is not 3.6, open IDLE of your python.
(2)Click File>Open...
(3)Find the document SimulationTool. Choose Main.py to open.
(4)Run Main.py


NOTE:There some things should be noticed when using the simulation software.
#NOTE 1: Once run the software, the original output files will be overwriten. So, if you want to keep the original outputs, please rename the files, or save them in another folder.
#NOTE 2: Make sure RESULT.Report.xlsx (output file) is not open when running the software.