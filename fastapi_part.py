from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from typing import List, Dict
import pickle
import sklearn
import pandas as pd
import csv

from utils import preprocess, extra_features, make_prediction, col_reorder

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str = None # optional field
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


'''
sample_row = Item(name = 'Mahindra Xylo E4 BS IV',
                  year = 2010,
                  selling_price = 229999,
                  km_driven = 168000,
                  fuel = 'Diesel', 
                  seller_type = 'Individual', 
                  transmission = 'Manual',
                  owner = 'First Owner', 
                  mileage = '14.0 kmpl', 
                  engine = '2498 CC',
                  max_power = '112 bhp', 
                  torque = '260 Nm at 1800-2200 rpm',
                  seats = 7.0
                  )
'''
                  

def item_to_df(item):
    test_df = item.model_dump() # create like a dict
    test_df = pd.DataFrame([test_df])
    return test_df

def items_to_df(items):
    test_df = pd.DataFrame([item.dict() for item in items])
    return test_df

def pipeline(df):

    df = preprocess(df)
    df = extra_features(df)
    df = col_reorder(df)
    df.to_csv('test_df.csv', index=False)

    output = make_prediction(df)

    return output




'''
test_df = item_to_df(sample_row)
output = pipeline(test_df)
print('Prediction: ' + str(output))
'''



def dataframe_to_pydantic(dataframe: pd.DataFrame, model: BaseModel) -> List[BaseModel]:
    df_dict = dataframe.to_dict(orient = 'records')
    #print(df_dict)
    items = []
    for row in df_dict:
        try:
            item = model(**row)
            items.append(item)
        except ValidationError as e:
            print(f"Validation error: {e}")

    return items




@app.post("/predict_item")
def predict_item(item: Item) -> float:
    input_df = item_to_df(item)
    prediction = pipeline(input_df)[0]
    return prediction


@app.post("/predict_items")
#def predict_items(items: List[Item]) -> List[float]:
def predict_items(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    #df_dict = df.to_dict(orient='dict')
    #print(df_dict)
    #pydantic_object = Items.model_validate(df_dict)
    data_items = dataframe_to_pydantic(df, Item)
    #print(data_items)

    input_df = items_to_df(data_items)
    print(input_df.columns)
    input_df['selling_price-est'] = pipeline(input_df)
    
    print(input_df)

    # save output as csv file (TEMP!!!)
    input_df.to_csv('input_df.csv', index=False)
    
    # Convert the DataFrame to CSV format as bytes
    csv_bytes = input_df.to_csv(index=False).encode('utf-8')

    # Create a StreamingResponse to stream the CSV file as a response
    return StreamingResponse(iter([csv_bytes]), media_type="text/csv", headers={"Content-Disposition": "attachment;filename=data.csv"})

   

@app.get("/")
def root():
    #return "Hello, student!"
    return {"message": "Hello, student!"}