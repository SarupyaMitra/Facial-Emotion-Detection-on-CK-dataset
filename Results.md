For the CK+ dataset, using 16(2 different scales \* 8 different orientations) different Gabor Filters and using 4\*4 blocks per output and using only the blockwise mean as the feature. **The different scales were '4' and '8'. The different orientations were pi/8 , pi/4 , 3\*pi/8 , pi/2 , 5\*pi/8 , 3\*pi/4 , 7\*pi/8 , pi.** 



**The result:** 



Accuracy: 0.8010752688172043

Confusion Matrix :

&nbsp;\[\[  3   0   0   0   2   0   4   0]

&nbsp;\[  1   7   0   0   0   0   4   0]

&nbsp;\[  0   0   3   1   0   0   1   0]

&nbsp;\[  0   0   1  11   0   1   0   1]

&nbsp;\[  0   0   0   0   2   0   4   0]

&nbsp;\[  0   0   0   0   1  14   2   0]

&nbsp;\[  0   0   0   1   5   1 109   3]

&nbsp;\[  0   0   1   0   0   1   2   0]]

Classification Report :

&nbsp;              precision    recall  f1-score   support



&nbsp;          0       0.75      0.33      0.46         9

&nbsp;          1       1.00      0.58      0.74        12

&nbsp;          2       0.60      0.60      0.60         5

&nbsp;          3       0.85      0.79      0.81        14

&nbsp;          4       0.20      0.33      0.25         6

&nbsp;          5       0.82      0.82      0.82        17

&nbsp;          6       0.87      0.92      0.89       119

&nbsp;          7       0.00      0.00      0.00         4



&nbsp;   accuracy                           0.80       186

&nbsp;  macro avg       0.64      0.55      0.57       186

weighted avg       0.82      0.80      0.80       186





**Changing the scales to '4' and '12', the result is:**



Accuracy: 0.7903225806451613

Confusion Matrix :

&nbsp;\[\[  2   0   0   0   2   0   5   0]

&nbsp;\[  0   8   0   0   0   0   4   0]

&nbsp;\[  0   0   3   1   0   0   1   0]

&nbsp;\[  0   0   1  11   0   1   0   1]

&nbsp;\[  0   0   0   0   2   0   4   0]

&nbsp;\[  1   0   0   0   1  13   2   0]

&nbsp;\[  0   1   1   0   5   1 108   3]

&nbsp;\[  0   0   1   0   0   1   2   0]]

Classification Report :

&nbsp;              precision    recall  f1-score   support



&nbsp;          0       0.67      0.22      0.33         9

&nbsp;          1       0.89      0.67      0.76        12

&nbsp;          2       0.50      0.60      0.55         5

&nbsp;          3       0.92      0.79      0.85        14

&nbsp;          4       0.20      0.33      0.25         6

&nbsp;          5       0.81      0.76      0.79        17

&nbsp;          6       0.86      0.91      0.88       119

&nbsp;          7       0.00      0.00      0.00         4



&nbsp;   accuracy                           0.79       186

&nbsp;  macro avg       0.61      0.54      0.55       186

weighted avg       0.80      0.79      0.79       186



**Increasing the number of different scales to 3 having values '4' , '8' , '12' and keeping the same orientations, the result is:** 

Accuracy: 0.7795698924731183

Confusion Matrix :

&nbsp;\[\[  3   0   0   0   2   0   4   0]

&nbsp;\[  0   6   0   0   0   0   6   0]

&nbsp;\[  0   0   3   1   0   0   1   0]

&nbsp;\[  1   0   1  11   0   0   0   1]

&nbsp;\[  1   0   0   0   2   0   3   0]

&nbsp;\[  1   0   0   0   1  13   2   0]

&nbsp;\[  0   2   0   1   5   1 107   3]

&nbsp;\[  0   0   1   0   0   2   1   0]]

Classification Report :

&nbsp;              precision    recall  f1-score   support



&nbsp;          0       0.50      0.33      0.40         9

&nbsp;          1       0.75      0.50      0.60        12

&nbsp;          2       0.60      0.60      0.60         5

&nbsp;          3       0.85      0.79      0.81        14

&nbsp;          4       0.20      0.33      0.25         6

&nbsp;          5       0.81      0.76      0.79        17

&nbsp;          6       0.86      0.90      0.88       119

&nbsp;          7       0.00      0.00      0.00         4



&nbsp;   accuracy                           0.78       186

&nbsp;  macro avg       0.57      0.53      0.54       186

weighted avg       0.79      0.78      0.78       186



