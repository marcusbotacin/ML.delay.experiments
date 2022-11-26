# Models

## model1.py
	* Word2vec, no drift detector
## model2.py
	* TF-IDF, no drift detector
## model3.py
	* Same word2vec as model1, but using adaptive RF
	* why performance is worse?

## model4.py
	* Dont remember what it does (hahaha)

## model5.py
	* First version that seems to work and statistics are OK (still no drift detector working)

## model7.py
	* Most complete version so far. Functional and with drift detection
	* With dataset forced balance.
	* Gave up using ARF. RF has worked best.
	* Still need to implement limited retraining, warning
	* Still to implement the queue

# Printing

## line.py
	* Line plot, 0-1 range, plot precision

## line2.py
	* Line plot, 0-100 range, plot exposure metric

## line3.py
	* Line plot, 2-axis, plot precision and exposure together

## compare_precision.py
	* Same as line.py, but two arguments

## compare_exposure.py
	* Same as line2.py, but two arguments
	
# Simulation

## pipeline.py
	* Mock code for the simulation experiments
