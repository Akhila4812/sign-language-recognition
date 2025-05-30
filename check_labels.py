import pandas as pd

# Load the dataset
train_df = pd.read_csv('C:/Users/akhil/sign_language/data/sign_mnist_train.csv')
test_df = pd.read_csv('C:/Users/akhil/sign_language/data/sign_mnist_test.csv')

# Print unique labels
print('Unique labels in training set:', sorted(train_df['label'].unique()))
print('Unique labels in test set:', sorted(test_df['label'].unique()))
