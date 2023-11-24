import pandas as pd
import pickle

DEBUG = False

def load_data(testset_file_path = './data/panic_disorder_dataset_testing.csv'):
  test_x = pd.read_csv(testset_file_path, na_values='NA')
  return test_x

def load_model_params(model_file_path = './models/best_model_prob.pkl',
                      trainset_columns_file_path = './models/columns.pkl',
                      scaler_file_path = './models/scaler.pkl'):
  # load model
  with open(model_file_path, 'rb') as f:
    model = pickle.load(f)

  # load columns
  with open(trainset_columns_file_path, 'rb') as f:
    columns = pickle.load(f)

  # load scaler
  with open(scaler_file_path, 'rb') as f:
    scaler = pickle.load(f)

  param = {
    'model': model,
    'columns': columns,
    'scaler': scaler
  }
  return param


def predict(test_x, param):

  # data preprocessing
  if DEBUG:
    test_x = test_x.drop('Panic Disorder Diagnosis', axis=1)
    test_x = test_x.drop('Participant ID', axis=1)

  test_x = pd.get_dummies(test_x, drop_first=True)

  # align testset columns with training set columns
  columns = param['columns']
  test_x = test_x.reindex(columns=columns, fill_value=0)

  # load scaler
  scaler = param['scaler']
  test_x = scaler.transform(test_x)

  # predict
  model = param['model']
  prediction = model.predict(test_x)

  return prediction

def prob_predict(test_x, param):
  test_x = pd.get_dummies(test_x, drop_first=True)

  # align testset columns with training set columns
  columns = param['columns']
  test_x = test_x.reindex(columns=columns, fill_value=0)

  # load scaler
  scaler = param['scaler']
  test_x = scaler.transform(test_x)

  # predict
  model = param['model']
  prob_prediction = model.predict_proba(test_x)

  return prob_prediction
  


if __name__ == '__main__':
  DEBUG = True
  
  print("Loading data...")
  test_x = load_data()
  
  print("Loading model params...")
  param = load_model_params()

  print("Predicting...")
  res = predict(test_x, param)
  prob = prob_predict(test_x, param)
  
  print(prob)