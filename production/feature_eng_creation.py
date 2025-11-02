import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import CountFrequencyEncoder
from feature_engine.transformation import LogTransformer 
from feature_engine.selection import DropFeatures

#cargamos operadores definidos por desarrollador
import logging
import operators

logging.basicConfig(filename="prod_ml_system.log", encoding="utf-8", filemode="a", level=logging.INFO,

                    format="{asctime}, {levelname}, {message}", style="{", datefmt="%Y-%m-%d %H:%M")

#imputacion de variables categoricas con imputación por frecuencia
CATEGORICAL_VARS_WITH_NA_FREQUENT=['BsmtQual','BsmtExposure','BsmtFinType1', 'GarageFinish', 'Functional',
                                   'MSZoning', 'Exterior1st', 'KitchenQual']

#'BsmtFullBath', 'GarageCars',

#Imputacion de variables númericas con imputacion por media
NUMERICAL_VARS_WITH_NA =['LotFrontage','GarageArea']

#Imputacion de variables categoricas con valor faltante (Missing)
CATEGORICAL_VARS_WITH_NA_MISSING=['FireplaceQu']

#Variables Temporales
TEMPORAL_VARS=['YearRemodAdd']

#Año de referencia
REF_VAR= "YrSold"

#Variables para binarizacion por sesgo
BINARIZE_VARS=['ScreenProch']

#Variables que eliminaremos
DROP_FEATURES = ["YrSold"]

#Variables para transfomraicón logarítmica
NUMERICAL_LOG_VARS = ["LotFrontage", "1stFlrSF", "GrLivArea"]

#Variables para codificación ordinal.
QUAL_VARS = ['ExterQual', 'BsmtQual', 'HeatingQC', 'KitchenQual', 'FireplaceQu']

 #variables especiales
EXPOSURE_VARS = ['BsmtExposure']
FINISH_VARS = ['BsmtFinType1']
GARAGE_VARS = ['GarageFinish']
FENCE_VARS = ['Fence']

#Variables para codificación por frecuencia (no ordinal)
CATEGORICAL_VARS = ['MSZoning',  'LotShape',  'LandContour', 'LotConfig', 'Neighborhood', 'RoofStyle', 
                    'Exterior1st', 'Foundation', 'CentralAir', 'Functional', 'PavedDrive', 'SaleCondition']

#Mapeo para varibles categótricas para calidad.
QUAL_MAPPINGS = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'Missing':0, 'NA':0}
EXPOSURE_MAPPINGS = {'No':1, 'Mn':2, 'Av':3, 'Gd':4}
FINISH_MAPPINGS = {'Missing':0, 'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}
GARAGE_MAPPINGS = {'Missing':0, 'NA':0, 'Unf':1, 'RFn':2, 'Fin':3}

#Variables a utilzar en el entrenamiento

FEATURES = ['MSSubClass','MSZoning','LotFrontage','LotShape','LandContour','LotConfig','Neighborhood',
            'RoofStyle','Exterior1st','ExterQual','Foundation','BsmtQual','BsmtExposure','BsmtFinType1',
            'HeatingQC','CentralAir','1stFlrSF','GrLivArea','BsmtFullBath','KitchenQual','Functional',
            'FireplaceQu','GarageFinish','GarageCars','GarageArea','PavedDrive','WoodDeckSF','SaleCondition']



def load_n_pre_data():
    data_train = pd.read_csv("../data/raw/train.csv") #para moverse de la carpeta colocamos ../ y vamos a traer la ruta
    data_train['MSSubClass'] = data_train['MSSubClass'].astype('O')
    data_train['GarageCars'] = data_train['GarageCars'].astype('O')
    data_train['BsmtFullBath'] = data_train['BsmtFullBath'].astype('O')

    #split en train test.
    X_train, X_test, y_train, y_test= train_test_split(data_train.drop(['Id','SalePrice'], axis=1), data_train['SalePrice'], test_size=0.30, random_state=2025)

    return X_train, y_train

def create_n_config_preproc_pipeline(X_train, y_train):
    ALL_FEATURES=set(X_train.columns)
    FEATURES_TO_DROP =ALL_FEATURES.difference(FEATURES)
    FEATURES_TO_DROP = list(FEATURES_TO_DROP)

    house_prices_data_pre_proc= Pipeline([
#0. Seleccion de features para el modelo 
    ('drop_features', DropFeatures(features_to_drop=FEATURES_TO_DROP)),
#1. Imputacion de variables categoricas
    ('cat_missing_imputation', CategoricalImputer(imputation_method='missing', variables=CATEGORICAL_VARS_WITH_NA_MISSING)),

#2. Imputacion de variables categoricas por frecuencia
    ('cat_missing_freq_imputation', CategoricalImputer(imputation_method='frequent', variables=CATEGORICAL_VARS_WITH_NA_FREQUENT)),

#3. Imputacion de variables númericas
    ('mean_imputation', MeanMedianImputer(imputation_method='mean', variables=NUMERICAL_VARS_WITH_NA)),

#4.Codificacion de variables categoricas
    ('quality_mapper', operators.Mapper(variables=QUAL_VARS, mappins=QUAL_MAPPINGS)),

    ('exposure_mapper', operators.Mapper(variables=EXPOSURE_VARS, mappins=EXPOSURE_MAPPINGS)),

    ('garage_mapper', operators.Mapper(variables=GARAGE_VARS, mappins=GARAGE_MAPPINGS)),

    ('Finish_mapper', operators.Mapper(variables=FINISH_VARS, mappins=FINISH_MAPPINGS)),

#5. Codificacion por Frecuency encoding
    ('cat_freq_encode', CountFrequencyEncoder(encoding_method='count', variables=CATEGORICAL_VARS)),

#6.Transformacion de variables continuas
    ('continues_log_transform', LogTransformer(variables=NUMERICAL_LOG_VARS)),

#7. Normalizacion de variables
    ('Variable_scaler', MinMaxScaler())
])
    
    house_prices_data_pre_proc.fit(X_train, y_train)
    joblib.dump(house_prices_data_pre_proc, '../models/house_prices_data_pre_proc_pipeline.pkl')

    return house_prices_data_pre_proc

def save_procesed_data(X,y, str_df_name, house_prices_data_pre_proc):
    X_transformed = house_prices_data_pre_proc.transform(X)
    df_X_train_transformed= pd.DataFrame(data=X_transformed, columns=FEATURES)
    y = y.reset_index()
    df_transformed= pd.concat([df_X_train_transformed, y['SalePrice']], axis=1)
    df_transformed.to_csv(f"../data/interim/proc_{str_df_name}.csv", index=False)


def main():
    try:
        logging.info("✅Iniciando Preprocesamiento de Datos")
        print("✅Iniciando Preprocesamiento de Datos")
        #cargamos y configuramos datos de entrada
        X_train, y_train = load_n_pre_data()

        logging.info("✅Datos Cargados y configurados correctamente")
        print("✅Datos Cargados y configurados correctamente")

        #creamos y configuramos el pipeline
        pipeline= create_n_config_preproc_pipeline(X_train, y_train)

        logging.info("✅Pipeline Creado y configurado correctamente")
        print("✅Pipeline Creado y configurado correctamente")


        #Guardamos datos preprocesador para el entrenamiento
        save_procesed_data(X_train, y_train, 'data_train', pipeline)
        logging.info("✅Datos de Entrenamiento Guardados Correctamente")
        print("✅Datos de Entrenamiento Guardados Correctamente")
    except Exception as ex:
        logging.error(f"⛔Error{ex}")
        print(f"⛔Error{ex}")



if __name__=="__main__":
    main()