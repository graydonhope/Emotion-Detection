# Emotion-Detection
Uses OpenCV and your front-facing camera to detect your emotion based on facial features. By accessing your live camera or webcam data, this model can detect some common emotions and send back a classification for live interaction. 

### Some examples
#### Here are some examples with different backgrounds and lighting.


![Smiling](https://user-images.githubusercontent.com/41659296/71372598-d94e8780-2582-11ea-8ce9-b7409005fd90.PNG)


#### Model can still classify with some disturbance (eg. your annoying cat trying to jump into your photos)
![no smile with noise](https://user-images.githubusercontent.com/41659296/71372618-e4091c80-2582-11ea-81df-5fb12086f2d7.PNG)


#### Glasses on
![Not smiling](https://user-images.githubusercontent.com/41659296/71372658-f71bec80-2582-11ea-904d-558e38195580.PNG)

#### Glasses off and hoodie on
![smile_with_noise](https://user-images.githubusercontent.com/41659296/71372754-42ce9600-2583-11ea-92f9-a1c9c0e54f25.PNG)

## Features
  • If you are smiling and you say 'Cheese!' the camera will snap and save a picture. This is a hands-free approach to getting high-quality and happy selfies.

  • Live feed classification. This model accesses the camera using OpenCV so it can classify emotions very quickly and return the results to the screen.
  
  • Still functional with different lighting and noise in picture (ex with cat or glasses on)
  
  
## Technical 
#### Using tensorboard to visualize some of the runs epochs

##### Epoch Accuracy
We can see that the number of epochs doesn't necessarily increase the accuracy. It is important to find a good balance for the number of epochs so that you aren't wasting time and resources on training the model when it isn't improving accuracy. 
![epoch accuracy](https://user-images.githubusercontent.com/41659296/71373507-672b7200-2585-11ea-81f2-cce869c8f761.PNG)


##### Epoch Loss
![epoch loss](https://user-images.githubusercontent.com/41659296/71373515-6a266280-2585-11ea-8e7c-de57d2dbf200.PNG)


#### Alternative Runs

![epoch accuracy 1](https://user-images.githubusercontent.com/41659296/71373521-6e528000-2585-11ea-9569-32d5b684787f.PNG)



![first_few-epochs_accuracy](https://user-images.githubusercontent.com/41659296/71373533-82967d00-2585-11ea-93b4-4d8694452c6a.PNG)



![first_few_epochs_loss](https://user-images.githubusercontent.com/41659296/71373539-8b874e80-2585-11ea-8f6f-ec25a084bb04.PNG)


![epochs_training](https://user-images.githubusercontent.com/41659296/71373544-917d2f80-2585-11ea-9b42-442c53fcf581.PNG)


##### Full Training Curves
![training_curves](https://user-images.githubusercontent.com/41659296/71373560-a22da580-2585-11ea-9840-d1eb299d4b6f.PNG)


#### Please note only the model is uploaded at this time. There are other supporting files used to interact with the camera and data which I am keeping private at this time. 
