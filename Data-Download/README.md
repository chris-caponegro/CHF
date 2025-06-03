This is a data set builder to help create a dataset of CHF and healthy patients for model training.

How it works:
The final Dataset will be stored in the Dataset folder. Not /Dataset is inside the .gitignore meaning it will never get pushed to github (too large anyway). 

Instead you will have to follow the python scripts inside the /scripts folder. The scripts include:

FindPleth: takes a Big Query csv and will determine if the patient has a Pleth wave. Stores results in Downloads/CHF_BQ_mimicX.csv

Download