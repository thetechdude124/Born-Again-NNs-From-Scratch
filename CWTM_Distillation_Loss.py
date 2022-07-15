import torch
from torch import nn
from torch.autograd import Function
from fastai.vision.all import *
#Import Distillation Test from Test_Distillations (runs on MNIST to verify that graidients are flowing smoothly)
from Test_Distillations import testDistillation

#CWTM (Confidence Weighted by Teacher Max) Distillation Loss - subclass of torch.autograd.Function, all methods are static
class CWTM_DistillationLoss(Function):

    #Define forward method (where we compute the loss) - take in the student predictions, teacher predictions, and true predictions for each model
    @staticmethod
    def forward(ctx, s_preds, t_preds, true_preds):
        #We want to save the NORMALIZED (softmax activated) versions of each tensor as opposed to the raw probabilities - normalize the values first, and then save
        softmax = nn.Softmax(dim = 1) #Perform across each row (each row sums to 1)
        #The true predictions are already encoded and do not requre softmax
        s_soft_preds = softmax(s_preds)
        t_soft_preds = softmax(t_preds)
        #Save both prediction tensors + true label tensor into context object for gradient computations
        ctx.save_for_backward(s_soft_preds, t_soft_preds, true_preds)
        #Convert the true predictions into a PyTorch tensor 
        true_preds = torch.tensor(true_preds)
        #Use Cross Entropy Loss with mean reduction to calculate differences between probability distributions rather than a distribution and labels
        loss_function = nn.CrossEntropyLoss(reduction = "mean")
        #Compute and return loss
        loss = loss_function(s_preds, t_preds)
        #Print tensor predictions for debugging
        #print('STUDENT PREDICTIONS: \n', s_preds)
        print('SOFTMAX STUDENT PREDICTIONS: \n', s_soft_preds)
        #print('TEACHER PREDICTIONS: \n', t_preds)
        print('SOFTMAX TEACHER PREDICTIONS: \n', t_soft_preds)
        print('TRUE PREDICTIONS: \n', true_preds)
        return loss

    #Define backward method (where the gradient of the loss is computed)
    @staticmethod
    def backward(ctx, grad_output):
        #Implement Equation 10 from the BAN paper https://proceedings.mlr.press/v80/furlanello18a/furlanello18a.pdf
        #Obtain labels from the saved tensors
        s_smax_preds, t_smax_preds, true_preds = ctx.saved_tensors
        #Find the probabilities of the predicted teacher classes, as well as the class predicted
        t_preds, t_pred_classes = torch.max(t_smax_preds, dim = 1)
        #Find the SUM of all the teacher probabilities within the minibatch
        t_label_sum = torch.sum(t_preds, dim = 0)
        #Divide each element in t_pred_labels by the total teacher sum
        weight_tensor = torch.divide(t_preds, t_label_sum)
        #Find the difference between the STUDENT probability distribution and the GROUND TRUTH probability distribution
        #One hot encode true predictions to do this
        one_hot_encoded_true_labels = torch.nn.functional.one_hot(true_preds, num_classes = s_smax_preds.shape[1])
        diff = torch.sub(s_smax_preds, one_hot_encoded_true_labels)
        #Convert to vector (remove extra dimension) to allow for element-wise multiplication
        weight_tensor.unsqueeze_(dim = 1)
        #Multiply the weight tensor by the gradients to get the final gradient update, normalize by batch size (first element in the tensor)
        batch_size = s_smax_preds.shape[0]
        print(weight_tensor.shape)
        print(diff.shape)
        grad_input = (1 / batch_size) * torch.matmul(diff, weight_tensor)
        print(grad_input.shape)
        print('GRADIENTS: \n', grad_input)
        #Return gradient to update student parameters - neither the teacher nor the true preds must have their gradients updated (return None)
        return grad_input, None, None

#If the script is run directly from the terminal, test the distillation loss/gradient on MNIST 
if __name__ == "__main__":
    testDistillation(CWTM_DistillationLoss, n_args = 3)