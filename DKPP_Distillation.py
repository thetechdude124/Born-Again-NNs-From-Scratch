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
        #In other words, shuffle the non-argmax logits of each sample
        #First get the maximum values and indices
        highest_pred, highest_pred_idxs = torch.max(t_smax_preds, dim = 1)
        #Unsqueeze for row-wise comparisons
        highest_pred.unsqueeze_(dim = 1)
        highest_pred_idxs.unsqueeze_(dim = 1)
        #Create a mask to remove all of the maximum values out of the tensor, leaving just logits that can be permuted
        remove_max_mask = t_smax_preds.scatter_(1, highest_pred_idxs, 0)
        teacher_nonmax_logits = t_smax_preds[remove_max_mask.bool()].view(batch_size, s_smax_preds.shape[1] - 1)
        #Permute leftover logits
        permuted_indexes = torch.randperm(teacher_nonmax_logits.shape[0])
        permuted_logits = teacher_nonmax_logits[permuted_indexes]
        #Append the argmax sample column back into the array (shape is now back bs * n_classes)
        argmax_added_logits = torch.cat((permuted_logits, highest_pred), dim = 1)
        #Create tensor of size (0, n_classes)
        permuted_DK_logits = torch.zeros(0, argmax_added_logits.shape[1])
        #Iterate over argmax_added_logits rows and place the argmax element back to its original position - concatenate each row to the above defined tensor
        for (row_idx, row), (sample_number, max_val), (sample_number, max_index) in zip(enumerate(argmax_added_logits), enumerate(highest_pred), enumerate(highest_pred_idxs)):
            #Get first part of the row (everything up to the original position of the argmax value), and concatenate with the argmax value
            first_section = torch.cat((row[:max_index.item()].clone().detach(), max_val), dim = 0)
            #As we appended all of the maximum values as a vector to the end of the logits matrix, the last element will always be the argmax one
            #So, go from the max index to one short of the row's end to get the remaining part of the row
            #Remember that the original row has not been modified - we are obtaining everything after and including the max index
            second_section = row[max_index.item():row.shape[0]-1].clone().detach()
            #Combine the two sections
            new_row = torch.cat((first_section, second_section), dim = 0)
            #Unsqueeze to turn into matrix and then add to permuted_DK_logits
            new_row.unsqueeze_(dim = 0)
            permuted_DK_logits = torch.cat((permuted_DK_logits, new_row), dim = 0)
        #Now that we have the permuted DK logits, subtract all of the NON-ARGMAX values with torch.where
        subtracted_logits = torch.where(permuted_DK_logits != highest_pred, s_smax_preds - permuted_DK_logits, permuted_DK_logits)
        #Normalize to obtain second term
        second_grad_term = (1 / batch_size) * subtracted_logits
        #Element wise - for each element, if the value is the max value for that sample, leave it unchanged
        #Otherwise randomly permute it (and the other non-argmax samples in that row) to serve as the logits for another sample
        #permuted_DK_terms = torch.where(t_smax_preds != highest_pred, t_smax_preds[permuted_indexes], t_smax_preds)
        #second_grad_term = (1 / batch_size) * torch.sub(s_smax_preds, permuted_DK_terms)
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