# Celsius to Farenhiet

let us assume the data we have is :-

| Celsius   | -40 | -10 | 0 | 8 | 15  | 22  | 38  |
|-----------|-----|-----|---|---|-----|-----|-----|
| Farenhiet | -40 | 14  | 32| 46| 59  | 72  | 100 |


## Algorithm approach:-
If you are familiar with the conversion of **clesius** to **Farenhiet** (i.e. c*1.8 + 32 = F)
then its pretty easy for you to solve this.
Other ways can be like using computational methods like lagerangean polynimals,..etc

## Machine Learning approach:-
We have our Input here as **Celsius** and our Output here is **Farenhiet**.
Now its our task to determine the rule to solve this.

Don't get confused, we are just finding the formula of conversion.

To achieve this lets create a simple neural network which will have single layer with one neuron.
This layer will get the input and adjust its weights accordingly.

**NOTE:-** If we see this neuron mathematically, then it will be something like, ans = input*w1 + b1
Here, w1 is the weight and b1 is the bias.
w1 only becuase only one parameter is given as input to this layer i.e. celsius.
b1 is some bias which will always be thereas single for each neuron.

The output of this layer(neuralnetwork coz of singlelayer) will be the farenhiet.

Wait, we are not done yet!
The ouput we obtain will not be our answer, we should optimize it inorder to get the output.

As we are using TensorFlow, so we use our optimizer as `Adam` optimizer and epoch value as 500.
Epoch value is nothing but the number of iterations that should be done to optimize our model for finest solution.

Let's code it and [check it out](https://github.com/Anirudh3167/ML_LEARNING/blob/main/celsius_to_farenhiet/single_layer_model.py)


### Method-2:-
we can make the single layered neural network as multi-layered.

Let's code it and [check it out](https://github.com/Anirudh3167/ML_LEARNING/blob/main/celsius_to_farenhiet/multi_layer_model.py)
