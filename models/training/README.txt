
Data Preparation: The syntax errors in the dataset were fixed using preparing.py, and the dataset was adapted for single classification.
	->The location and function columns were merged into a single column called location_function, resulting in 16 distinct classes instead of 8*2.
	->The classes were coded numerically from 0 to 15.


Model Training: The dataset was sent to the tokenizer and model using gpt_training.py, and the training process was carried out.
	->Training and evaluation processes were handled using functions found in util.py.


Model Testing: Using test_model.py, the saved model weights and tokenizer were used to send example sentences to the model.
	->The class predictions made by the model were mapped back to the location_function structure, and the predicted function was correctly identified.