# Accelerometer-identification

The problem I am going to solve consist on identifying different persons (specifically 22 persons) by using acceleromenter data.

The dataset provided consists of accelerometer data of 22 unique, walking individuals, collected using a smartphone device. For every individual there are multiple recorded fragments of them walking, with variable durations. For some people there are more fragments than the others. In total there are 1857 walking-fragments, consisting of x, y and z acceleration values per unit of time, labeled with their relevant person ID. A sampling frequency of 50 Hz was considered when collecting each of the acceleration values that conform a given fragment.


The Main code, together with a more detailed explanation of the analysis performed and the model trained in this work can be found in the notebook:

https://github.com/Riccocez/Accelerometer-identification/blob/master/Main.ipynb

As requested, the predictions from the test data given can be found in the file y_pred.json, which is contained in this repository.
