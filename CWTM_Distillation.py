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
        true_preds_T = torch.tensor(true_preds)
        #As PyTorch does not have a satisfactory way of computing cross entropy between two distributions (as opposed to labels), implement it from scratch
        loss = - torch.sum(torch.mul(t_soft_preds, torch.log(s_soft_preds))).mean() + 10e-10 #Add small constant to prevent zeroes
        #This computes the cross entropy between student and LABEL predictions rather than with the teacher
        #The purpose of this is to verify whether the student is "learning" in terms of the original dataset (i.e. whether both the student-teacher and student-label losses are decreasing)
        true_labels_loss_func = nn.CrossEntropyLoss()
        true_labels_loss = true_labels_loss_func(s_preds, true_preds_T)
        print('TRUE LABEL CROSSENTROPY LOSS: ', true_labels_loss)
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
        one_hot_encoded_true_labels = nn.functional.one_hot(true_preds, num_classes = s_smax_preds.shape[1])
        diff = torch.sub(s_smax_preds, one_hot_encoded_true_labels)
        #Convert to vector (remove extra dimension) to allow for element-wise multiplication
        weight_tensor.unsqueeze_(dim = 1)
        #Multiply the weight tensor by the gradients to get the final gradient update, normalize by batch size (first element in the tensor)
        #Take the "element-wise" product of the weight tensor and difference (vector and matrix) to preserve matrix dims for gradient
        batch_size = s_smax_preds.shape[0]
        grad_input = (1 / batch_size) * (weight_tensor * diff)
        #print('GRADIENTS: \n', grad_input)
        #Return gradient to update student parameters - neither the teacher nor the true preds must have their gradients updated (return None)
        return grad_input, None, None

#If the script is run directly from the terminal, test the distillation loss/gradient on MNIST via the testDistillation function from Test_Distillations.py
if __name__ == "__main__":
    testDistillation(CWTM_DistillationLoss)