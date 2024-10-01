import pandas as pd
import pickle
import numpy as np
import json
from geopy.distance import geodesic
class PredictDelivery(object):
    def __init__(self):
        self.city_label_encoder                 = pickle.load(open('parameter/City_label_encoder.pkl','rb'))
        self.day_of_week_scaler                 = pickle.load(open('parameter/day_of_week_scaler.pkl','rb'))
        self.day_scaler                         = pickle.load(open('parameter/day_scaler.pkl','rb'))
        self.Delivery_person_Age_scaler         = pickle.load(open('parameter/Delivery_person_Age_scaler.pkl','rb'))
        self.Delivery_person_Ratings_scaler     = pickle.load(open('parameter/Delivery_person_Ratings_scaler.pkl','rb'))
        self.distance_scaler                    = pickle.load(open('parameter/distance_scaler.pkl','rb'))
        self.Festival_label_encoder             = pickle.load(open('parameter/Festival_label_encoder.pkl','rb'))
        self.month_scaler                       = pickle.load(open('parameter/month_scaler.pkl','rb'))
        self.multiple_deliveries_scaler         = pickle.load(open('parameter/multiple_deliveries_scaler.pkl','rb'))
        self.order_prepare_time_scaler          = pickle.load(open('parameter/order_prepare_time_scaler.pkl','rb'))
        self.quarter_scaler                     = pickle.load(open('parameter/quarter_scaler.pkl','rb'))
        self.Type_of_order_label_encoder        = pickle.load(open('parameter/Type_of_order_label_encoder.pkl','rb'))
        self.Type_of_vehicle_label_encoder      = pickle.load(open('parameter/Type_of_vehicle_label_encoder.pkl','rb'))
        self.Vehicle_condition_scaler           = pickle.load(open('parameter/Vehicle_condition_scaler.pkl','rb'))
        self.Weatherconditions_label_encoder    = pickle.load(open('parameter/Weatherconditions_label_encoder.pkl','rb'))

    def data_formatation(self, df):
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        df['Time_Orderd'] = pd.to_datetime(df['Time_Orderd'], errors='coerce')
        df['Time_Order_picked'] = pd.to_datetime(df['Time_Order_picked'], errors='coerce')
        df['Type_of_order'] = df['Type_of_order'].str.strip()
        df['Type_of_vehicle'] = df['Type_of_vehicle'].str.strip()
        df['Weatherconditions'] = df['Weatherconditions'].str.strip()
        return df

    
    def calculate_time_diff(self, df):
        time_diff = df['Time_Order_picked'] - df['Time_Orderd']

        # Corrigindo os valores negativos (caso haja diferenças que cruzam dias)
        time_diff = time_diff.apply(lambda x: x + pd.Timedelta(days=1) if x.days < 0 else x)

        # Convertendo a diferença para minutos
        df['order_prepare_time'] = time_diff.dt.total_seconds() / 60

        df['order_prepare_time'].fillna(df['order_prepare_time'].median(), inplace=True)
        return df
    
    def calculate_distance(self, df):
        df['distance'] = np.zeros(len(df))
        restaurante_coordinates = df[['Restaurant_latitude', 'Restaurant_longitude']].to_numpy()
        cliente_coordinates = df[['Delivery_location_latitude', 'Delivery_location_longitude']].to_numpy()
        df['distance'] = np.array([geodesic(restaurant, delivery) for restaurant, delivery in zip(restaurante_coordinates, cliente_coordinates)])
        df['distance'] = df['distance'].astype('str').str.extract('(\d+)').astype('int64')
        return df

    def feature_engineering(self, df2):

        df2['day'] = df2.Order_Date.dt.day
        df2['month'] = df2.Order_Date.dt.month
        df2['quarter'] = df2.Order_Date.dt.quarter
        df2['year'] = df2.Order_Date.dt.year
        df2['day_of_week'] = df2.Order_Date.dt.day_of_week.astype(int)
        df2['is_month_start'] = df2.Order_Date.dt.is_month_start.astype(int)
        df2['is_month_end'] = df2.Order_Date.dt.is_month_end.astype(int)
        df2['is_quarter_start'] = df2.Order_Date.dt.is_quarter_start.astype(int)
        df2['is_quarter_end'] = df2.Order_Date.dt.is_quarter_end.astype(int)
        df2['is_year_start'] = df2.Order_Date.dt.is_year_start.astype(int)
        df2['is_year_end'] = df2.Order_Date.dt.is_year_end.astype(int)
        df2['is_weekend'] = np.where(df2['day_of_week'].isin([5,6]),1,0)
        df2 = self.calculate_time_diff(df2)
        df2 = self.calculate_distance(df2)

        return df2
    
    def data_preparation(self, df3):
        df3['day_of_week']                 = self.day_of_week_scaler.transform(df3[['day_of_week']])
        df3['day']                         = self.day_scaler.transform(df3[['day']])
        df3['Delivery_person_Age']         = self.Delivery_person_Age_scaler.transform(df3[['Delivery_person_Age']])
        df3['Delivery_person_Ratings']     = self.Delivery_person_Ratings_scaler.transform(df3[['Delivery_person_Ratings']])
        df3['distance']                    = self.distance_scaler.transform(df3[['distance']])
        df3['month']                       = self.month_scaler.transform(df3[['month']])
        df3['multiple_deliveries']         = self.multiple_deliveries_scaler.transform(df3[['multiple_deliveries']])
        df3['order_prepare_time']          = self.order_prepare_time_scaler.transform(df3[['order_prepare_time']])
        df3['quarter']                     = self.quarter_scaler.transform(df3[['quarter']])
        df3['Vehicle_condition']           = self.Vehicle_condition_scaler.transform(df3[['Vehicle_condition']])
        

        
        ordem_categoria = {'Low':1, 'Medium':2, 'High':3, 'Jam':4}
        df3['Road_traffic_density'] = df3['Road_traffic_density'].map(ordem_categoria)
        df3['City']                 = self.city_label_encoder.transform(df3[['City']])
        df3['Festival']             = self.Festival_label_encoder.transform(df3[['Festival']])
        
        df3['Type_of_order']        = self.Type_of_order_label_encoder.transform(df3[['Type_of_order']])
        df3['Type_of_vehicle']      = self.Type_of_vehicle_label_encoder.transform(df3[['Type_of_vehicle']])
        df3['Weatherconditions']    = self.Weatherconditions_label_encoder.transform(df3[['Weatherconditions']])
        

        cols_selected_boruta = ['Delivery_person_Age',
                                'Delivery_person_Ratings',
                                'Weatherconditions',
                                'Road_traffic_density',
                                'Vehicle_condition',
                                'multiple_deliveries',
                                'distance']

        return df3[cols_selected_boruta]

    def get_predictions(self, model, test_data, original_data):
        pred = model.predict(test_data)
        original_data['prediction'] = pred

        return original_data.to_json(orient='records')
