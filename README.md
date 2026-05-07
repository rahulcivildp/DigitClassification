* Download Miniconda from [https://www.anaconda.com/download/success](https://www.anaconda.com/docs/getting-started/main)

* Install conda acording to the system os.

* Then run:
conda --version

* Search in Windows start menu:
Anaconda Prompt

* Inside Anaconda Prompt, run:
conda init powershell

* Open VS Code, new terminal:
conda activate
conda create -n digit python=3.14
conda activate digit

* Create requirements.txt to include missing packages
nano requirements.txt

* Then type: 
matplotlib
scikit-learn

* Install all packages in requirements.txt
pip install -r requirements.txt

* Run:
python plot_digits_classification.py
