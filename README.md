# Wine data analysis

Data taken from https://www.kaggle.com/zynicide/wine-reviews

This short project analyses data made of wine descriptions provided by wine reviewers, points given, price and other properties like location and variety. 
The project consists of:
- analysis of numerical variables, concerned with exploring distribtuons and identifying any correlations
- applying nlp library `spacy` to extract each wine flavor profile and food pairings, if present
- interactive dashboard, where one can tune parameteres (like country, price etc) and see different descriptive plots as well as information about wine flavors and pairings
### Dashborads examples:
#### Example 1
Shows barchart of quatities of selected vines per country.
![image](https://user-images.githubusercontent.com/54853811/189065868-f93583d1-dc65-48ee-9313-44bb45798efb.png)

#### Example 2
A wordcloud of most frequent flavours for given vine settings and main flavor categories ordered by popularity. Also depicts a flavor profile and info of randomly picked vine (from sample with settings from control panel).
![image](https://user-images.githubusercontent.com/54853811/189065952-adb39ac4-77a0-446d-8718-bf8990970ef2.png)

#### Example 3
Suggests food pairings for given category of vines ordered by popularity. Also shows food pairing and info for a randomly picked vine from given sample.
![image](https://user-images.githubusercontent.com/54853811/189066006-0b708d0c-3410-4f57-89cd-11d851275fcc.png)
