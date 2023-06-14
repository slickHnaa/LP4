import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('store_sales_regressor.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_cat = data["le_cat"]
num_scalar = data["num_scalar"]

def show_predict_page():
    st.title("Favorita Stores Sales Prediction Predict Page")

    st.write("""### Plesae provide the following information""")


    store_nbr = [25,  1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  2, 23, 24, 26, 27,
       28,  3, 30, 31, 32, 33, 34, 35, 37, 38, 39,  4, 40, 41, 43, 44, 45,
       46, 47, 48, 49,  5, 50, 51, 54,  6,  7,  8,  9, 36, 53, 20, 29, 21,
       42, 22, 52]

    onpromotion = [  0,   2,   1,   4,   6,   3,  35,  19,  39,  27, 114,   7, 118,
        42,  25,  44,  37,  24,  14,  40,   5,  26,  20,  38, 126,  21,
        10,  15,  36,   9, 137,  52,  12,   8, 136,  51,  45,  31,  46,
        11,  28,  16,  23,  43,  22, 144, 142,  18,  34, 138,  17, 151,
        32,  56,  30,  50, 145,  41,  58,  13,  29,  49, 146,  33, 141,
        47, 147,  59,  65, 186,  62,  55,  54,  67,  81,  63,  91,  53,
        76,  94, 105, 170, 102,  66, 116,  61, 112,  68,  64, 157, 156,
       174, 185, 184, 197, 199, 202, 215, 204, 210,  48, 206,  57, 203,
       201, 213, 217, 190,  71,  83,  90,  88,  93,  78, 117, 125,  80,
       208,  75,  60,  69,  70,  74,  98, 104,  73, 107, 115, 113,  95,
        77,  82, 121, 119, 106, 129, 123, 103,  85, 133, 109,  89, 110,
        84, 135, 139, 108,  87, 128, 120, 152, 132,  86, 134,  99, 101,
        92,  96,  72, 111, 100,  79,  97, 198, 191, 486, 122, 130, 127,
       131, 143, 154, 155, 171, 167, 187, 196, 214, 207, 211, 221, 223,
       124, 209, 153, 161, 169, 172, 180, 179, 182, 181, 212, 189, 178,
       200, 216, 163, 183, 195, 192, 194, 166, 176, 193, 205, 218, 150,
       407, 158, 165, 173, 148, 140, 162, 159, 164, 177, 168, 160, 175,
       222, 226, 228, 236, 149, 231, 229, 188, 219, 227, 220, 235, 225,
       224, 230, 233, 244, 243, 245, 261, 489, 476, 446, 435, 470, 507,
       474, 383, 444, 452, 464, 473, 234, 240, 330, 232, 697, 289, 609,
       304, 639, 299, 630, 655, 286, 633, 302, 644, 237, 626, 317, 624,
       672, 306, 600, 293, 258, 646, 305, 642, 276, 290, 269, 716]

    oil_price = [ 93.14,  97.01,  97.48,  97.1 ,  91.23,  94.09,  90.74,  93.84,
        95.25, 101.92, 107.13, 105.41, 105.47, 106.61, 107.43, 103.07,
       101.63, 102.17,  94.74,  94.25,  95.13,  93.12,  97.14,  98.62,
        98.87,  99.18,  98.17,  95.14, 105.34, 103.64,  99.69, 104.05,
       104.35, 100.89, 107.2 , 107.95, 106.83, 107.04, 106.07, 106.06,
       104.76, 104.19, 104.06, 102.93, 103.81, 102.76, 105.23,  97.34,
        97.3 ,  88.89,  85.76,  85.87,  78.77,  77.87,  78.71,  77.43,
        77.85,  77.16,  65.94,  68.98,  65.89,  63.13,  55.25,  56.78,
        55.7 ,  54.59,  53.45,  52.72,  53.56,  49.59,  50.12,  52.08,
        53.3 ,  55.58,  59.1 ,  60.72,  61.05,  59.59,  52.48,  48.11,
        47.98,  45.13,  44.94,  38.22,  44.4 ,  47.86,  49.67,  46.12,
        47.88,  44.32,  44.23,  42.95,  41.74,  40.57,  40.43,  37.46,
        34.55,  36.12,  36.76,  37.62,  37.13,  29.71,  27.96,  34.57,
        37.99,  35.36,  42.12,  41.45,  39.74,  40.88,  42.72,  43.18,
        42.76,  41.67,  42.52,  45.29,  46.03,  45.98,  44.75,  43.65,
        43.77,  44.33,  44.58,  43.45,  44.68,  46.21,  46.64,  46.22,
        47.72,  48.04,  49.36,  49.34,  42.4 ,  41.83,  41.75,  44.47,
        45.72,  45.32,  44.66,  44.07,  44.88,  44.62,  43.39,  46.72,
        45.66,  51.72,  50.95,  50.84,  51.44,  51.98,  52.01,  52.82,
        52.36,  54.04,  54.  ,  52.63,  53.12,  53.19,  52.62,  49.64,
        48.83,  47.83,  50.99,  49.58,  42.86,  45.11,  47.77,  48.54,
        48.81,  47.57,  46.29,  47.07,  49.76]

    cluster = [ 1,  2,  3,   4,  5,  6,  7, 8,  9, 10, 11, 12, 13,  14, 15, 16, 17]

    transactions = [50, 573, 892, 342, 214, 125, 863]
    city = ['Salinas', 'Quito', 'Cayambe', 'Latacunga', 'Riobamba', 'Ibarra',
       'Santo Domingo', 'Guaranda', 'Ambato', 'Guayaquil', 'Daule',
       'Babahoyo', 'Quevedo', 'Playas', 'Cuenca', 'Loja', 'Machala',
       'Esmeraldas', 'El Carmen', 'Libertad', 'Manta', 'Puyo']

    family = ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS',
       'BREAD/BAKERY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS',
       'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE',
       'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES',
       'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE',
       'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 'PERSONAL CARE',
       'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY',
       'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES',
       'SEAFOOD']

    holiday_type = ['Holiday', 'Additional', 'Transfer', 'Event', 'Bridge']

    Family = st.selectbox("What's Family of product", family)
    Holiday = st.selectbox("Is day a Holiday", holiday_type)
    City = st.selectbox('City', city)
    Cluster = st.selectbox('Select a cluster where of the store', cluster)
    Oil_price = st.selectbox('Enter the price of Oil for that day', oil_price)
    Transactions = st.selectbox('What is the value of transactions', transactions)
    Onpromotion = st.selectbox('onpromotion', onpromotion)
    Store_nbr = st.selectbox('store_nbr', store_nbr)




    ok = st.button("Predict")
    if ok:
        X = np.array([[City, Family, Holiday, Store_nbr, Onpromotion, Transactions, Oil_price, Cluster]])
        X[:, 0] = le_cat.transform([X[0, 0]])[0]
        #st.write(X[:, 0])
        X[:, 1] = le_cat.transform([X[0, 1]])[0]
        #st.write(X[:, 1])
        X[:, 2] = le_cat.transform([X[0, 2]])[0]
        #st.write(X[:, 2])
        X[:, 3] = num_scalar.transform([X[0, 3]])[0]
        #st.write(X[:, 3])
        X[:, 4] = num_scalar.transform([X[0, 4]])[0]
        #st.write(X[:, 4])
        X[:, 5] = num_scalar.transform([X[0, 5]])[0]
        #st.write(X[:, 5])
        X[:, 6] = num_scalar.transform([X[0, 6]])[0]
        #st.write(X[:, 6])
        X[:, 7] = num_scalar.transform([X[0, 7]])[0]
        #st.write(X[:, 7])
        X = X.astype(float)
        #st.write(X)
        Sales = regressor.predict(X)
        st.subheader(f"The estimated sale is ${Sales[0]:.2f}")
