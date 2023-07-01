from sklearn.preprocessing import StandardScaler

#Function to normalize the data
def dataNormalize(ndarray):
    scaler = StandardScaler() 
    data_scaled = scaler.fit_transform(ndarray)
    return data_scaled
