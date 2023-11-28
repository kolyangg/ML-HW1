import re
import pandas as pd
import pickle
import numpy as np

columns_to_convert = ['mileage', 'engine', 'max_power']
object_columns = ['owner', 'fuel', 'seller_type', 'transmission', 'seats']
non_num_columns_X = ['name', 'selling_price', 'fuel', 'seller_type', 'transmission', 'owner', 'seats'] # ADDED SEATS

model_columns_backup = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque',
       'max_torque_rpm', 'transmission_Manual', 'seller_type_Individual',
       'seller_type_Trustmark Dealer', 'fuel_Diesel', 'fuel_LPG',
       'fuel_Petrol', 'owner_Fourth & Above Owner', 'owner_Second Owner',
       'owner_Test Drive Car', 'owner_Third Owner', 'seats_4', 'seats_5',
       'seats_6', 'seats_7', 'seats_8', 'seats_9', 'seats_10', 'seats_14',
       'max_power_perl', 'name_Audi', 'name_BMW', 'name_Chevrolet',
       'name_Daewoo', 'name_Datsun', 'name_Fiat', 'name_Force', 'name_Ford',
       'name_Honda', 'name_Hyundai', 'name_Isuzu', 'name_Jaguar', 'name_Jeep',
       'name_Kia', 'name_Land', 'name_Lexus', 'name_MG', 'name_Mahindra',
       'name_Maruti', 'name_Mercedes-Benz', 'name_Mitsubishi', 'name_Nissan',
       'name_Peugeot', 'name_Renault', 'name_Skoda', 'name_Tata',
       'name_Toyota', 'name_Volkswagen', 'name_Volvo', 'third_owner_above',
       'one-two_owner_and_official', 'owners_number']


model_columns0 = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque',
       'max_torque_rpm', 'seller_type_Individual',
       'seller_type_Trustmark Dealer', 'fuel_Diesel', 'fuel_LPG',
       'fuel_Petrol', 'owner_Fourth & Above Owner', 'owner_Second Owner',
       'owner_Test Drive Car', 'owner_Third Owner', 'transmission_Manual',
       'seats_4', 'seats_5', 'seats_6', 'seats_7', 'seats_8', 'seats_9',
       'seats_10', 'seats_14', 'max_power_perl', 'name_Audi', 'name_BMW',
       'name_Chevrolet', 'name_Daewoo', 'name_Datsun', 'name_Fiat',
       'name_Force', 'name_Ford', 'name_Honda', 'name_Hyundai', 'name_Isuzu',
       'name_Jaguar', 'name_Jeep', 'name_Kia', 'name_Land', 'name_Lexus',
       'name_MG', 'name_Mahindra', 'name_Maruti', 'name_Mercedes-Benz',
       'name_Mitsubishi', 'name_Nissan', 'name_Peugeot', 'name_Renault',
       'name_Skoda', 'name_Tata', 'name_Toyota', 'name_Volkswagen',
       'name_Volvo', 'third_owner_above', 'one-two_owner_and_official',
       'owners_number']

model_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 
                 'owner_Fourth & Above Owner', 'owner_Second Owner', 'owner_Test Drive Car', 'owner_Third Owner', 
                 'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol', 'seller_type_Individual', 'seller_type_Trustmark Dealer', 
                 'transmission_Manual', 'seats_4', 'seats_5', 'seats_6', 'seats_7', 'seats_8', 'seats_9', 'seats_10', 
                 'seats_14', 'max_power_perl', 'name_Ambassador Classic', 'name_Ambassador Grand', 'name_Audi A3', 'name_Audi A4', 
                 'name_Audi A6', 'name_Audi Q3', 'name_Audi Q5', 'name_Audi Q7', 'name_BMW 3', 'name_BMW 5', 'name_BMW 6', 'name_BMW 7', 
                 'name_BMW X1', 'name_BMW X3', 'name_BMW X4', 'name_BMW X6', 'name_BMW X7', 'name_Benz B', 'name_Benz CLA', 'name_Benz E', 
                 'name_Benz GL', 'name_Benz GLA', 'name_Benz GLC', 'name_Benz M', 'name_Benz New', 'name_Benz S', 'name_Chevrolet Aveo', 
                 'name_Chevrolet Beat', 'name_Chevrolet Captiva', 'name_Chevrolet Cruze', 'name_Chevrolet Enjoy', 'name_Chevrolet Optra', 
                 'name_Chevrolet Sail', 'name_Chevrolet Spark', 'name_Chevrolet Tavera', 'name_Chevrolet Trailblazer', 'name_Daewoo Matiz', 
                 'name_Datsun GO', 'name_Datsun RediGO', 'name_Fiat Avventura', 'name_Fiat Grande', 'name_Fiat Linea', 'name_Fiat Palio', 
                 'name_Fiat Punto', 'name_Force Gurkha', 'name_Force One', 'name_Ford Aspire', 'name_Ford Classic', 'name_Ford EcoSport', 
                 'name_Ford Ecosport', 'name_Ford Endeavour', 'name_Ford Fiesta', 'name_Ford Figo', 'name_Ford Freestyle', 'name_Ford Fusion', 
                 'name_Ford Ikon', 'name_Honda Accord', 'name_Honda Amaze', 'name_Honda BR', 'name_Honda BRV', 'name_Honda Brio', 'name_Honda CR', 
                 'name_Honda City', 'name_Honda Civic', 'name_Honda Jazz', 'name_Honda Mobilio', 'name_Honda WR', 'name_Hyundai Accent', 
                 'name_Hyundai Creta', 'name_Hyundai EON', 'name_Hyundai Elantra', 'name_Hyundai Elite', 'name_Hyundai Getz', 'name_Hyundai Grand', 
                 'name_Hyundai Santa', 'name_Hyundai Santro', 'name_Hyundai Sonata', 'name_Hyundai Tucson', 'name_Hyundai Venue', 'name_Hyundai Verna', 
                 'name_Hyundai Xcent', 'name_Hyundai i10', 'name_Hyundai i20', 'name_Isuzu D', 'name_Isuzu MU', 'name_Isuzu MUX', 'name_Jaguar XE', 'name_Jaguar XF', 
                 'name_Jeep Compass', 'name_Jeep Wrangler', 'name_Kia Seltos', 'name_Land Rover', 'name_Lexus ES', 'name_MG Hector', 'name_Mahindra Bolero', 
                 'name_Mahindra Ingenio', 'name_Mahindra Jeep', 'name_Mahindra KUV', 'name_Mahindra Logan', 'name_Mahindra Marazzo', 'name_Mahindra Marshal', 
                 'name_Mahindra NuvoSport', 'name_Mahindra Quanto', 'name_Mahindra Renault', 'name_Mahindra Scorpio', 'name_Mahindra Ssangyong', 'name_Mahindra Supro', 
                 'name_Mahindra TUV', 'name_Mahindra Thar', 'name_Mahindra Verito', 'name_Mahindra Willys', 'name_Mahindra XUV300', 'name_Mahindra XUV500', 
                 'name_Mahindra Xylo', 'name_Maruti 800', 'name_Maruti A', 'name_Maruti Alto', 'name_Maruti Baleno', 'name_Maruti Celerio', 'name_Maruti Ciaz', 
                 'name_Maruti Dzire', 'name_Maruti Eeco', 'name_Maruti Ertiga', 'name_Maruti Esteem', 'name_Maruti Estilo', 'name_Maruti Gypsy', 'name_Maruti Ignis', 
                 'name_Maruti Omni', 'name_Maruti Ritz', 'name_Maruti S', 'name_Maruti SX4', 'name_Maruti Swift', 'name_Maruti Vitara', 'name_Maruti Wagon', 
                 'name_Maruti XL6', 'name_Maruti Zen', 'name_Mitsubishi Lancer', 'name_Mitsubishi Pajero', 'name_Nissan Kicks', 'name_Nissan Micra', 'name_Nissan Sunny', 
                 'name_Nissan Teana', 'name_Nissan Terrano', 'name_Peugeot 309', 'name_Renault Captur', 'name_Renault Duster', 'name_Renault Fluence', 'name_Renault KWID', 
                 'name_Renault Koleos', 'name_Renault Lodgy', 'name_Renault Pulse', 'name_Renault Scala', 'name_Renault Triber', 'name_Skoda Fabia', 'name_Skoda Laura', 
                 'name_Skoda Octavia', 'name_Skoda Rapid', 'name_Skoda Superb', 'name_Skoda Yeti', 'name_Tata Aria', 'name_Tata Bolt', 'name_Tata Estate', 'name_Tata Harrier', 
                 'name_Tata Hexa', 'name_Tata Indica', 'name_Tata Indigo', 'name_Tata Manza', 'name_Tata Nano', 'name_Tata New', 'name_Tata Nexon', 'name_Tata Safari', 
                 'name_Tata Spacio', 'name_Tata Sumo', 'name_Tata Tiago', 'name_Tata Tigor', 'name_Tata Venture', 'name_Tata Winger', 'name_Tata Xenon', 'name_Tata Zest', 
                 'name_Toyota Camry', 'name_Toyota Corolla', 'name_Toyota Etios', 'name_Toyota Fortuner', 'name_Toyota Glanza', 'name_Toyota Innova', 'name_Toyota Land', 
                 'name_Toyota Platinum', 'name_Toyota Premio', 'name_Toyota Qualis', 'name_Toyota Yaris', 'name_Volkswagen Ameo', 'name_Volkswagen CrossPolo', 'name_Volkswagen GTI', 
                 'name_Volkswagen Jetta', 'name_Volkswagen Multivan', 'name_Volkswagen Passat', 'name_Volkswagen Polo', 'name_Volkswagen Vento', 'name_Volvo S60', 'name_Volvo S90', 
                 'name_Volvo V40', 'name_Volvo XC40', 'name_Volvo XC90', 'third_owner_above', 'one-two_owner_and_official', 'owners_number']

def extract1(text):
  A = list(filter(bool, re.findall(r'[0-9.]*', text)))
  return A[0] if A else None

def extract2(text):
  A = list(filter(bool, re.findall(r'[0-9.]*', text)))
  return A[-1] if A else None


def apply_scaler(df):
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    df_num = df.drop(columns = non_num_columns_X) # drop object columns
    df_num = pd.DataFrame(scaler.transform(df_num), index=df_num.index, columns=df_num.columns)
    df = pd.concat([df[non_num_columns_X], df_num], axis=1) # verify_integrity - means keep values from second df

    return df


def apply_ohe(df):
    with open('ohe.pkl', 'rb') as file:
        ohe = pickle.load(file)
    df.to_csv('test_df_ohe.csv', index=False)

    df_cat_columns = pd.DataFrame(ohe.transform(df[object_columns]).toarray(), columns = ohe.get_feature_names_out())
    df = pd.concat([df, df_cat_columns], axis=1)
    df = df.drop(columns = object_columns) # drop original columns

    return df


def apply_ohe_name(df):
    with open('ohe_name.pkl', 'rb') as file:
        ohe_name = pickle.load(file)
    
    df_cat_columns = pd.DataFrame(ohe_name.transform(df[['name']]).toarray(), columns = ohe_name.get_feature_names_out())
    df = pd.concat([df, df_cat_columns], axis=1)
    df = df.drop(columns = 'name') # drop original columns

    return df



def preprocess(df):
    df[columns_to_convert] = df[columns_to_convert].apply(lambda x: x.replace('[^0-9.]', '', regex = True))
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    df[columns_to_convert] = df[columns_to_convert].astype(float)

    df['torque'] = df['torque'].str.replace(',', '')
    df['torque'] = df['torque'].astype(str)

    df['torque_old'] = df['torque']

    df['torque'] = df['torque_old'].apply(lambda x: extract1(x))
    df['max_torque_rpm'] = df['torque_old'].apply(lambda x: extract2(x))
    df['torque'] = df['torque'].astype(float) # CHECK!!!
    df['max_torque_rpm'] = df['max_torque_rpm'].astype(float) # CHECK!!!
    
    df = df.drop(columns=['torque_old'])

    # apply scaler
    df = apply_scaler(df)

    # apply OHE
    df = apply_ohe(df)


    return df   

def extra_features(df):
    df['max_power_perl'] = df['max_power'] / df['engine']
    df['year'] = df['year'] ** 2
    df['name'] = df['name'].str.extract(r'(\w+\s\w+)') # add OHE name

    # apply OHE name
    df = apply_ohe_name(df)

    df['third_owner_above'] = ((df['owner_Third Owner'] == 1) | (df['owner_Fourth & Above Owner'] == 1)).astype(int)
    df['one-two_owner_and_official'] = ((df['owner_Third Owner'] == 1) & (df['owner_Fourth & Above Owner'] == 0) & (df['seller_type_Trustmark Dealer'] == 1)).astype(int)

    conditions = [
        (df['owner_Test Drive Car'] == 1) | (df['owner_Second Owner'] == 0) & (df['owner_Third Owner'] == 0) & (df['owner_Fourth & Above Owner'] == 0),
        (df['owner_Second Owner'] == 1),
        (df['owner_Third Owner'] == 1),
        (df['owner_Fourth & Above Owner'] == 1)
    ]

    choices = [1, 2, 3, 4]

    df['owners_number'] = np.select(conditions, choices, default=1).astype(int)


    return df

def col_reorder(df):
    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    df = df[model_columns]


    return df


def make_prediction(df):
    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    #prediction = model.predict(df)
    prediction = model.predict(df)

    return prediction