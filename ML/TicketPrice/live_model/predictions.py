import joblib
import kagglehub
import modin.pandas as pd

# Load dataset (example: historical ticket pricing data)
path = kagglehub.dataset_download("ibrahimelsayed182/plane-ticket-price")
print("Path to dataset files:", path)
data_path = fr'{path}\ticket_pricing_data.csv'
raw = pd.read_csv(r'C:\Users\Alex\.cache\kagglehub\datasets\ibrahimelsayed182\plane-ticket-price\versions\1\Data_Train.csv')

# Load the label encoders and the scaler
label_encoders = joblib.load(r'C:\Projects\DataSci\ML\TicketPrice\live_model\model_outputs\label_encoders.pkl')
scaler = joblib.load(r'C:\Projects\DataSci\ML\TicketPrice\live_model\model_outputs\scaler.pkl')

#use encoders and scaler to transform the input data
def preprocess_data(raw):
    raw = raw.dropna()
    X = raw.drop('Price', axis=1)
    y = raw['Price']
    X = X.apply(lambda x: label_encoders[x.name].transform(x))
    X = scaler.transform(X)
    return X, y

X, y = preprocess_data(raw)

# Load the models
model_1 = joblib.load(r'C:\Projects\DataSci\ML\TicketPrice\live_model\model_outputs\meta_model.pkl')

# Make predictions
predictions = model_1.predict(X)
print(predictions)