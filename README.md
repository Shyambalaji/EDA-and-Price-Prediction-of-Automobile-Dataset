# EDA-and-Price-Prediction-of-Automobile-Dataset

Steps in data analysis and prediction of price :
     
       1.The data is first read in the form of csv
       2.The data is then checked for any null values or any unfilled values and it is replaced with mean or median.
       3.The univariate plot,the correlation plot and the plot between the attributes is done using the respective libraries.
       4.The final step is the prediction of price by selecting the attributes which are highly related to the price attribute than other attributes.
         the data set is divded for train and test in the ratio of 8:2.

Supervised learning model used : Multiple linear regression  

Attributes which has stronger relationship with price -
1. Curb-Weight
2. Engine-Size
3. Horsepower
4. Mpg(City / Highway mpg)
5. Lenght/ Width

Chevy which is a brand of General motors had the highest milage followed by the Japanese car makers.
European car makers except Volkswagen which sells Luxary cars.So the Mileage of European car makers are lower.
Cars with lower engine capacity generally have higher fuel economy.

1.DHC (Direct overhead cam) tyoe of engines are more in the data.
2.Most cars sold have 4 doors
3.Petrol(Gas) cars are more popular.
4.Sedan Cars are most popular.

As expected Toyota sold more cars.the other car companies which sold more no of gars next to Toyoto are Mazda and Nissan.

Engine Size: It is the amount of air that can be sucked in by the engine.Generally it is measured in litres.
             For example an average car in India would have an engine capacity in the rane of 1-1.5 liter.

Price : In US today the median price of the vehicle is around 35000$.
        This is a old data so it shows very low median car price.

Curb weight : Is the total weight of the vehicle without the weight of the passenger.It includes weight of coolants,oil and fuel.Defination of curb weight may vary based on the standard adopted by a country.
              In this data set the curb weight of most cars is in the range 2000-3100 lbs.  
