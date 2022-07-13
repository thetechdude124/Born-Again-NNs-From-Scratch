import torch
from torch.autograd import Function
#Import the testing function from CWTM_Distillation.py to verify that the function works
from CWTM_Distillation_Loss import test

#DKPP (Dark Knowledge with Permuted Predictions) Distillation Loss - purpose is to test if the Dark Knowledge terms in Distillation Loss
#Are truly critical via permuting the dark knowledge logits so as to destroy the covariance matrix between max and 
class DKPP_DistillationLoss(Function):

    #The only required arguments for DKPP loss are the student and teacher predictions - the ground truth is not used as no importance weighting is done
    #The purpose is to test the Dark Knowledge Hypothesis, meaning that the student should be able to learn entirely form the DK passed on from the teacher
    @staticmethod
    def forward(ctx, s_preds, t_preds):
        #Save both tensors (softmax activated) into context for use in backward method
        softmax = torch.nn.Softmax(dim = 1)
        ctx.save_for_backward(softmax(s_preds), softmax(t_preds))
        #Loss between both is, as with the CWTM Distillation, the Binary Cross Entropy Loss (with Logits)
        loss = torch.nn.BCEWithLogitsLoss()
        return loss(s_preds, t_preds)

    @staticmethod
    def backward(ctx, grad_output):
        #Implementing equation 11 from the BAN paper https://proceedings.mlr.press/v80/furlanello18a/furlanello18a.pdf
        #Compute the gradients as usual (and scale based on batch size), but subtract via gradients of permuted teacher predictions
        #First, draw tensors from the context object
        s_smax_preds, t_smax_preds = ctx.saved_tensors
        #Find the true predicted label probabilities for both tensors (maximum of both probability distributions) - store returned index tensor in idx (disacrd)
        s_label_preds, idx = torch.max(s_smax_preds, dim = 1)
        t_label_preds, idx = torch.max(t_smax_preds, dim = 1)
        #Subtract both tensors and normalize by 1/b (where b is batch size) to obtain first term of gradient expression
        batch_size = s_label_preds.shape[0]
        first_grad_term = (1 / batch_size) * torch.subtract(s_label_preds, t_label_preds)
        #Next, subtract the DK student logits by the permuted DK teacher logits and normalize once again to obtain second gradient term
        #Access the indexes of the first dimension (the index of the rows) and use .randperm to shuffle indexes
        #Then, simply re-arrange the tensor with the new indexes to obtain the shuffled tensor
        permuted_indexes = torch.randperm(t_smax_preds.shape[0])
        permuted_DK_terms = t_smax_preds[permuted_indexes]
        second_grad_term = (1 / batch_size) * torch.subtract(s_smax_preds, permuted_DK_terms)
        #Before adding the terms together, repeat the first gradient term (it is currently a vector as the only the max predictions, as is typical for distillation losses) was considered
        #The Dark Knowledge term on the other hand contains information regarding all logits, so it must be a matrix of size batch * n_classes
        first_grad_term.unsqueeze_(dim = 1)
        first_grad_term = first_grad_term.repeat((1, s_smax_preds.shape[1]))
        #Add the first and second terms together and return as grad_input
        grad_input = torch.add(first_grad_term, second_grad_term)
        #No gradients needed for the teacher labels
        return grad_input, None

#If script is run from terminal, run the testing script (test functino found in CWTM_Distillation_Loss.py)
if __name__ == "__main__":
    test(DKPP_DistillationLoss, n_args = 2)