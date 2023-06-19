from django.shortcuts import render
import urllib
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB

def home(request):
    weather_df = pd.read_csv(r'D:\8thsem\Weather-Detection\datas.csv')
    isNull = weather_df.isnull().any()

    weather_df_num=weather_df.loc[:,['maxtempC','mintempC','cloudcover','humidity','tempC', 'sunHour', 'precipMM', 'pressure','windspeedKmph']]

    # Plotting all column values
    weather_df_num.plot(subplots=True, figsize=(25,20))
    # Convert the plot to a PNG image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    # Encode the PNG image to base64 to embed it in HTML
    allColumn = urllib.parse.quote(base64.b64encode(image_png))
    # #########

    # Histogram
    weather_df_num.hist(bins=10,figsize=(15,15))
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    histogram = urllib.parse.quote(base64.b64encode(image_png))
    
    
    weather_y=weather_df_num.pop("tempC")
    weather_x=weather_df_num
    train_X,test_X,train_y,test_y=train_test_split(weather_x,weather_y,test_size=0.2,random_state=4)

    # Multiple linear regression
    model=LinearRegression()
    model.fit(train_X,train_y)
    prediction = model.predict(test_X)
    result_df_mlr = pd.DataFrame({'Actual': test_y, 'Prediction': prediction, 'diff': (test_y - prediction)})
    result_subset_mlr = result_df_mlr.iloc[:10, :]
    # Convert the subset DataFrame to HTML table
    mlr = result_subset_mlr.to_html(index=False)
    #calculating error
    mae_mlr = np.mean(np.absolute(prediction-test_y))
    r2_mlr = r2_score(test_y,prediction ) 

    # Decision Tree Regression
    regressor=DecisionTreeRegressor(random_state=0)
    regressor.fit(train_X,train_y)
    prediction2=regressor.predict(test_X)
    result_df_dtr = pd.DataFrame({'Actual':test_y,'Prediction':prediction2,'diff':(test_y-prediction2)})
    result_subset_dtr = result_df_dtr.iloc[:10, :]
    dtr = result_subset_dtr.to_html(index=False)
    #calculating error
    mae_dtr = np.mean(np.absolute(prediction2-test_y))
    r2_dtr = r2_score(test_y,prediction2 ) 

    # Random Forest Regression
    regr=RandomForestRegressor(max_depth=90,random_state=0,n_estimators=100)
    regr.fit(train_X,train_y)
    prediction3=regr.predict(test_X)
    result_df_rfr = pd.DataFrame({'Actual':test_y,'Prediction':prediction3,'diff':(test_y-prediction3)})
    result_subset_rfr = result_df_rfr.iloc[:10, :]
    rfr = result_subset_rfr.to_html(index=False)
    #calculating error
    mae_rfr = np.mean(np.absolute(prediction3-test_y))
    r2_rfr = r2_score(test_y,prediction3 ) 

    # Gaussian Naive Bayes
    gnb=GaussianNB()
    gnb.fit(train_X,train_y)
    prediction4=gnb.predict(test_X)
    result_df_gnb = pd.DataFrame({'Actual':test_y,'Prediction':prediction4,'diff':(test_y-prediction4)})
    result_subset_gnb = result_df_gnb.iloc[:10, :]
    gnb = result_subset_gnb.to_html(index=False)
    #calculating error
    mae_gnb = np.mean(np.absolute(prediction4-test_y))
    r2_gnb = r2_score(test_y,prediction4 ) 
        
    # Pass the graphic variable to the template context
    context = {'allColumn': allColumn, 'histogram':histogram,'mlr':mlr,'dtr':dtr, 'rfr':rfr, 'gnb':gnb,'mae_mlr':mae_mlr,'r2_mlr':r2_mlr,'mae_dtr':mae_dtr,'r2_dtr':r2_dtr,'mae_rfr':mae_rfr,'r2_rfr':r2_rfr,'mae_gnb':mae_gnb,'r2_gnb':r2_gnb}
    # Render the template
    return render(request, 'home.html', context)
    
def predict(request):
    if request.method == 'POST':
        weather_df = pd.read_csv(r'D:\8thsem\Weather-Detection\datas.csv')
        isNull = weather_df.isnull().any()
        weather_df_num=weather_df.loc[:,['maxtempC','mintempC','cloudcover','humidity','tempC', 'sunHour', 'precipMM', 'pressure','windspeedKmph']]
        weather_y=weather_df_num.pop("tempC")
        weather_x=weather_df_num
        train_X,test_X,train_y,test_y=train_test_split(weather_x,weather_y,test_size=0.2,random_state=4)

        weather_df_num=weather_df.loc[:,['maxtempC','mintempC','cloudcover','humidity','tempC', 'sunHour', 'precipMM', 'pressure','windspeedKmph']]
        n1 = int(request.POST.get('n1'))
        n2 = int(request.POST.get('n2'))
        n3 = int(request.POST.get('n3'))
        n4 = int(request.POST.get('n4'))
        n5 = int(request.POST.get('n5'))
        n6 = int(request.POST.get('n6'))
        n7 = int(request.POST.get('n7'))
        n8 = int(request.POST.get('n8'))
        n9 = request.POST.get('algo')
        
        if(n9 == "mlr"):
            model=LinearRegression()
            model.fit(train_X,train_y)
            prediction = model.predict([[n1, n2, n3, n4, n5, n6, n7, n8]])
            context = {"result":prediction}
            return render(request, 'predict.html',context)
        if(n9 == "dtr"):
            regressor=DecisionTreeRegressor(random_state=0)
            regressor.fit(train_X,train_y)
            prediction2=float(regressor.predict([[n1, n2, n3, n4, n5, n6, n7, n8]]))
            context = {"result":prediction2}
            return render(request, 'predict.html',context)
        if(n9 == "rfr"):
            regr=RandomForestRegressor(max_depth=90,random_state=0,n_estimators=100)
            regr.fit(train_X,train_y)
            prediction3=regr.predict([[n1, n2, n3, n4, n5, n6, n7, n8]])
            context = {"result":prediction3}
            return render(request, 'predict.html',context)
        if(n9 == "gnb"):
            gnb=GaussianNB()
            gnb.fit(train_X,train_y)
            prediction4=gnb.predict([[n1, n2, n3, n4, n5, n6, n7, n8]])
            print("prediction",prediction4)
            context = {"result":prediction4}
            return render(request, 'predict.html',context)
    
    
    result = ""
    context = {"result":result}
    return render(request, 'predict.html',context)
    
