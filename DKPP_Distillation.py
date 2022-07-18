import torch
import torch.nn as nn
from torch.autograd import Function
#Import the testing function from Test_Distillations to verify that the function works
from Test_Distillations import testDistillation

#DKPP (Dark Knowledge with Permuted Predictions) Distillation Loss - purpose is to test if the Dark Knowledge terms in Distillation Loss
#are truly critical via permuting the dark knowledge logits so as to destroy the covariance matrix between the logits and true (maximum) predicted label
class DKPP_DistillationLoss(Function):

    #The only required arguments for DKPP loss are the student and teacher predictions - the ground truth is not used as no importance weighting is done
    #The purpose is to test the Dark Knowledge Hypothesis, meaning that the student should be able to learn entirely form the DK passed on from the teacher
    @staticmethod
    def forward(ctx, s_preds, t_preds, true_labels):
        #Save both tensors (softmax activated) into context for use in backward method
        softmax = nn.Softmax(dim = 1)
        s_soft_preds = softmax(s_preds)
        t_soft_preds = softmax(t_preds)
        ctx.save_for_backward(s_preds, t_preds, true_labels)
        #Same logic as with CWTM loss - compute cross entropy between distributions (implemented from scratch due to lack of PyTorch functionality)
        loss = - torch.sum(torch.mul(t_soft_preds, torch.log(s_soft_preds))).mean() + 10e-10 #small constant to prevent zeroes
        #Print CE against true labels (akin to CWTM forward method) for comparison
        true_labels_loss_func = nn.CrossEntropyLoss()
        true_labels_loss = true_labels_loss_func(s_preds, torch.tensor(true_labels))
        print('TRUE LABEL CROSSENTROPY LOSS: ', true_labels_loss)
        # print('STUDENT PREDICTIONS: \n', s_soft_preds)
        # print('TEACHER PREDICTIONS: \n', t_soft_preds)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        #Implementing equation 11 from the BAN paper https://proceedings.mlr.press/v80/furlanello18a/furlanello18a.pdf
        #Compute the gradients as usual (and scale based on batch size), but subtract via gradients of permuted teacher predictions
        #First, draw tensors from the context object
        s_smax_preds, t_smax_preds, true_labels = ctx.saved_tensors
        #Find the probability of the correct class for the student model, and the maximum probability per sample for the teacher model
        #For student - use true_labels as list of indicies and use gather to find values at those parts (unsqueeze first, and squeeze again to eliminate singleton dimension)
        true_labels.unsqueeze_(1)
        s_label_preds = torch.gather(s_smax_preds, dim = 1, index = true_labels)
        s_label_preds.squeeze_()
        t_label_preds, idxs = torch.max(t_smax_preds, dim = 1)
        #Subtract both tensors and normalize by 1/b (where b is batch size) to obtain first term of gradient expression
        batch_size = s_label_preds.shape[0]
        first_grad_term = (1 / batch_size) * torch.sub(s_label_preds, t_label_preds)
        #Next, subtract the DK student logits by the permuted DK teacher logits (EXCEPT for the argmax dimension - do not modify) and normalize once again 
        # to obtain second gradient term
        #Access the indexes of the first dimension (the index of the rows) and use .randperm to shuffle indexes
        #Then, simply re-arrange the tensor with the new indexes to obtain the shuffled tensor
        permuted_indexes = torch.randperm(t_smax_preds.shape[0])
        permuted_DK_terms = t_smax_preds[permuted_indexes]
        second_grad_term = (1 / batch_size) * torch.sub(s_smax_preds, permuted_DK_terms)
        #The first grad term is a vector of size bs, whereas the second grad term is of the size bs * n_classes - unsqueeze first term to (bs * 1)
        #As we must return bs * n_classes, add the vector to each column of the second grad term
        #Reverse order -> m + v instead of v + m
        first_grad_term.unsqueeze_(dim = 1)
        grad_input = torch.add(first_grad_term, second_grad_term)
        #No gradients needed for the teacher or true labels, return None 
        return grad_input, None, None

#If script is run from terminal, run the testing script (test functino found in CWTM_Distillation_Loss.py)
if __name__ == "__main__":
    testDistillation(DKPP_DistillationLoss)