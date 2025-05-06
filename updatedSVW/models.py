from svwFuncs import *


"""
do one pass for see vs watch 
second for which one
"""
# len embedd, num classes
model = nn.Sequential(nn.Linear(400, 7), nn.LogSoftmax(dim=1))

# 0.01

def trainingLoop(data, lr, ):
    # Define the loss
    criterion = nn.NLLLoss()
    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.SGD(model.parameters(), lr=lr)
    epochs = 10
    for e in range(epochs):
        running_loss = 0
        for featureVector, info, gold in data:
            optimizer.zero_grad() # empty the gradients, otherwise gradients are accumulated.
            output = model(featureVector)
            loss = criterion(output, gold)
            loss.backward() # auto-grad 
            optimizer.step() # update  weights 
            running_loss += loss.item()
            # else:
            #     print(f"Training epoch {e} : loss: {running_loss/len(trainloader)}")

    return None 



